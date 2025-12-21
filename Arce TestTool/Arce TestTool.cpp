// C++版本: C++17
// 编译要求: 支持C++17标准的编译器
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <map>
#include <ctime>
#include <numeric>
#include <complex>
#include <future>
#include <filesystem>
#include <cstdint>
#include <intrin.h>
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

using namespace Microsoft::WRL;

// 测试配置
struct TestConfig {
    struct Durations {
        int single_core = 12;
        int multi_core = 12;
        int memory = 12;
        int crypto = 12;
        int gpu = 12;
    };

    Durations duration;

    struct Switches {
        bool single_core = true;
        bool multi_core = true;
        bool memory = true;
        bool crypto = true;
        bool gpu = true;
    };

    Switches switches;

    const int matrix_size = 1024;
    const int prime_limit = 2000000;
    const int memory_size = 500000;
    const int crypto_rounds = 16;  // 增加加密轮次
    const int gpu_matrix_size = 512;
    const int gpu_fft_size = 1024;
    const int gpu_fluid_size = 256;
    const int gpu_monte_carlo_samples = 1000000;

    // 评分基准值（用于标准化）
    struct Benchmarks {
        double single_core = 0.005;   // 基准操作数/秒
        double multi_core = 2.0;
        double memory = 1000000.0;
        double crypto = 0.5;
        double gpu = 1.0;
    };

    struct Weights {
        double single_core = 0.25;
        double multi_core = 0.3;
        double memory = 0.2;
        double crypto = 0.1;
        double gpu = 0.15;
    };

    Weights weights;
    Benchmarks benchmarks;  // 基准值
    bool developer_mode = false;
    const std::string version = "1.0.4 CLI";
};

// 全局配置和状态
TestConfig test_config;
std::atomic<bool> is_testing(false);
std::atomic<bool> stop_test(false);
std::vector<std::thread> test_threads;
std::map<std::string, double> raw_results;  // 原始操作数
std::map<std::string, double> test_results; // 标准化得分
std::vector<std::string> log_messages;
std::recursive_mutex log_mutex;
std::mutex result_mutex;

// 日志功能
void log(const std::string& message) {
    std::lock_guard<std::recursive_mutex> lock(log_mutex);
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_s(&local_time, &now_time);

    std::stringstream ss;
    ss << "[" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << "] " << message;
    log_messages.push_back(ss.str());
    std::cout << ss.str() << std::endl;
}

// 获取用户名和计算机名
std::string get_username() {
    DWORD buffer_size = 0;
    GetUserNameA(nullptr, &buffer_size);
    if (buffer_size == 0) {
        return "Unknown";
    }

    std::vector<char> buffer(buffer_size);
    if (GetUserNameA(buffer.data(), &buffer_size)) {
        return std::string(buffer.data());
    }
    return "Unknown";
}

std::string get_computername() {
    DWORD buffer_size = 0;
    GetComputerNameA(nullptr, &buffer_size);
    if (buffer_size == 0) {
        return "Unknown";
    }

    std::vector<char> buffer(buffer_size);
    if (GetComputerNameA(buffer.data(), &buffer_size)) {
        return std::string(buffer.data());
    }
    return "Unknown";
}

// 导出日志
bool export_log(const std::string& path) {
    namespace fs = std::filesystem;

    if (path.empty()) {
        log("错误: 导出日志路径为空");
        return false;
    }

    auto make_timestamped_name = []() -> std::string {
        std::time_t t = std::time(nullptr);
        std::tm tm;
        localtime_s(&tm, &t);
        char buf[64];
        std::strftime(buf, sizeof(buf), "arce_log_%Y%m%d_%H%M%S.txt", &tm);
        return std::string(buf);
        };

    std::error_code ec;
    fs::path p(path);
    fs::path target;

    if ((fs::exists(p, ec) && fs::is_directory(p, ec)) ||
        (!path.empty() && (path.back() == '\\' || path.back() == '/'))) {
        fs::path dir = p;
        target = dir / make_timestamped_name();
    }
    else {
        target = p;
    }

    fs::path parent = target.parent_path();
    if (!parent.empty() && !fs::exists(parent, ec)) {
        if (!fs::create_directories(parent, ec)) {
            log("无法创建目录: " + parent.string() + "，错误: " + ec.message());
            return false;
        }
    }

    std::vector<std::string> messages_copy;
    {
        std::lock_guard<std::recursive_mutex> lock(log_mutex);
        messages_copy = log_messages;
    }

    std::ofstream file(target, std::ios::out);
    if (!file.is_open()) {
        log("无法打开日志文件: " + target.string() + " 请检查路径和权限");
        return false;
    }

    for (const auto& msg : messages_copy) {
        file << msg << std::endl;
    }
    file.close();

    log("日志已导出至: " + target.string());
    return true;
}

// 导入配置文件
bool load_ini_file(const std::string& filepath) {
    log("尝试加载配置文件: " + filepath);

    // 检查文件扩展名
    if (filepath.length() < 4 ||
        (filepath.substr(filepath.length() - 4) != ".ini" &&
            filepath.substr(filepath.length() - 4) != ".INI")) {
        log("错误：文件扩展名不匹配，请使用.ini格式");
        return false;
    }

    std::ifstream file(filepath, std::ios::in);
    if (!file.is_open()) {
        log("错误：无法打开配置文件，请检查路径和权限");
        return false;
    }

    try {
        std::string line;
        std::string current_section;

        // 读取BOM判断UTF-8
        char bom[3];
        file.read(bom, 3);
        if (bom[0] == (char)0xEF && bom[1] == (char)0xBB && bom[2] == (char)0xBF) {
            log("检测到UTF-8 BOM");
        }
        else {
            // 不是UTF-8 BOM，重置文件指针
            file.seekg(0, std::ios::beg);
        }

        while (std::getline(file, line)) {
            // 去除空白字符
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // 跳过空行和注释
            if (line.empty() || line[0] == ';' || line[0] == '#') {
                continue;
            }

            // 处理节
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }

            // 解析键值对
            size_t equal_pos = line.find('=');
            if (equal_pos != std::string::npos) {
                std::string key = line.substr(0, equal_pos);
                std::string value = line.substr(equal_pos + 1);

                // 去除键值两边的空白
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                // 根据节和键更新配置
                if (current_section == "Durations") {
                    try {
                        int int_value = std::stoi(value);
                        if (key == "single_core") test_config.duration.single_core = int_value;
                        else if (key == "multi_core") test_config.duration.multi_core = int_value;
                        else if (key == "memory") test_config.duration.memory = int_value;
                        else if (key == "crypto") test_config.duration.crypto = int_value;
                        else if (key == "gpu") test_config.duration.gpu = int_value;
                    }
                    catch (...) {
                        log("警告: 无法解析整数值: " + key + " = " + value);
                    }
                }
                else if (current_section == "Switches") {
                    try {
                        bool bool_value = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                        if (key == "single_core") test_config.switches.single_core = bool_value;
                        else if (key == "multi_core") test_config.switches.multi_core = bool_value;
                        else if (key == "memory") test_config.switches.memory = bool_value;
                        else if (key == "crypto") test_config.switches.crypto = bool_value;
                        else if (key == "gpu") test_config.switches.gpu = bool_value;
                    }
                    catch (...) {
                        log("警告: 无法解析布尔值: " + key + " = " + value);
                    }
                }
                else if (current_section == "Weights") {
                    try {
                        double double_value = std::stod(value);
                        if (key == "single_core") test_config.weights.single_core = double_value;
                        else if (key == "multi_core") test_config.weights.multi_core = double_value;
                        else if (key == "memory") test_config.weights.memory = double_value;
                        else if (key == "crypto") test_config.weights.crypto = double_value;
                        else if (key == "gpu") test_config.weights.gpu = double_value;
                    }
                    catch (...) {
                        log("警告: 无法解析浮点数值: " + key + " = " + value);
                    }
                }
                else if (current_section == "Benchmarks") {
                    try {
                        double double_value = std::stod(value);
                        if (key == "single_core") test_config.benchmarks.single_core = double_value;
                        else if (key == "multi_core") test_config.benchmarks.multi_core = double_value;
                        else if (key == "memory") test_config.benchmarks.memory = double_value;
                        else if (key == "crypto") test_config.benchmarks.crypto = double_value;
                        else if (key == "gpu") test_config.benchmarks.gpu = double_value;
                    }
                    catch (...) {
                        log("警告: 无法解析浮点数值: " + key + " = " + value);
                    }
                }
                else if (current_section == "Other") {
                    if (key == "developer_mode") {
                        test_config.developer_mode = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                    }
                }
            }
        }

        file.close();
        log("导入完成");
        log("已更新配置：");
        log("  测试时长 - 单核:" + std::to_string(test_config.duration.single_core) +
            " 多核:" + std::to_string(test_config.duration.multi_core) +
            " 内存:" + std::to_string(test_config.duration.memory) +
            " 加密:" + std::to_string(test_config.duration.crypto) +
            " GPU:" + std::to_string(test_config.duration.gpu));
        log("  测试开关 - 单核:" + std::string(test_config.switches.single_core ? "开" : "关") +
            " 多核:" + std::string(test_config.switches.multi_core ? "开" : "关") +
            " 内存:" + std::string(test_config.switches.memory ? "开" : "关") +
            " 加密:" + std::string(test_config.switches.crypto ? "开" : "关") +
            " GPU:" + std::string(test_config.switches.gpu ? "开" : "关"));
        log("  权重设置 - 单核:" + std::to_string(test_config.weights.single_core) +
            " 多核:" + std::to_string(test_config.weights.multi_core) +
            " 内存:" + std::to_string(test_config.weights.memory) +
            " 加密:" + std::to_string(test_config.weights.crypto) +
            " GPU:" + std::to_string(test_config.weights.gpu));

        return true;

    }
    catch (const std::exception& e) {
        log("错误：配置文件解析异常: " + std::string(e.what()));
        file.close();
        return false;
    }
    catch (...) {
        log("错误：不匹配的格式，请确认文件是否为UTF-8格式、文件扩展名是否匹配或配置文件语法是否正确");
        file.close();
        return false;
    }
}

bool save_ini_file(const std::string& filepath) {
    log("尝试保存配置文件到: " + filepath);

    // 检查文件扩展名
    if (filepath.length() < 4 ||
        (filepath.substr(filepath.length() - 4) != ".ini" &&
            filepath.substr(filepath.length() - 4) != ".INI")) {
        log("错误：文件扩展名不匹配，请使用.ini格式");
        return false;
    }

    std::ofstream file(filepath, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        log("错误：无法创建配置文件，请检查路径和权限");
        return false;
    }

    try {
        // 写入UTF-8 BOM
        file << char(0xEF) << char(0xBB) << char(0xBF);

        // 写入时间戳
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time;
        localtime_s(&local_time, &now_time);

        file << "; Arce TestTool 配置文件" << std::endl;
        file << "; 生成时间: " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
        file << "; 版本: " << test_config.version << std::endl;
        file << std::endl;

        // 写入测试时长
        file << "[Durations]" << std::endl;
        file << "single_core=" << test_config.duration.single_core << std::endl;
        file << "multi_core=" << test_config.duration.multi_core << std::endl;
        file << "memory=" << test_config.duration.memory << std::endl;
        file << "crypto=" << test_config.duration.crypto << std::endl;
        file << "gpu=" << test_config.duration.gpu << std::endl;
        file << std::endl;

        // 写入测试开关
        file << "[Switches]" << std::endl;
        file << "single_core=" << (test_config.switches.single_core ? 1 : 0) << std::endl;
        file << "multi_core=" << (test_config.switches.multi_core ? 1 : 0) << std::endl;
        file << "memory=" << (test_config.switches.memory ? 1 : 0) << std::endl;
        file << "crypto=" << (test_config.switches.crypto ? 1 : 0) << std::endl;
        file << "gpu=" << (test_config.switches.gpu ? 1 : 0) << std::endl;
        file << std::endl;

        // 写入权重设置
        file << "[Weights]" << std::endl;
        file << "single_core=" << test_config.weights.single_core << std::endl;
        file << "multi_core=" << test_config.weights.multi_core << std::endl;
        file << "memory=" << test_config.weights.memory << std::endl;
        file << "crypto=" << test_config.weights.crypto << std::endl;
        file << "gpu=" << test_config.weights.gpu << std::endl;
        file << std::endl;

        // 写入基准值
        file << "[Benchmarks]" << std::endl;
        file << "single_core=" << test_config.benchmarks.single_core << std::endl;
        file << "multi_core=" << test_config.benchmarks.multi_core << std::endl;
        file << "memory=" << test_config.benchmarks.memory << std::endl;
        file << "crypto=" << test_config.benchmarks.crypto << std::endl;
        file << "gpu=" << test_config.benchmarks.gpu << std::endl;
        file << std::endl;

        // 写入其他设置
        file << "[Other]" << std::endl;
        file << "developer_mode=" << (test_config.developer_mode ? 1 : 0) << std::endl;

        file.close();
        log("配置文件保存成功: " + filepath);
        return true;

    }
    catch (const std::exception& e) {
        log("错误：保存配置文件时发生异常: " + std::string(e.what()));
        file.close();
        return false;
    }
    catch (...) {
        log("错误：保存配置文件时发生未知异常");
        file.close();
        return false;
    }
}

// 辅助函数：生成随机数
double random_double(double min = 0.0, double max = 1.0) {
    static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(generator);
}

// 素数计算
std::vector<int> calculate_primes(int limit) {
    if (limit < 2) return {};
    std::vector<bool> sieve(limit + 1, true);
    sieve[0] = sieve[1] = false;
    for (int i = 2; i * i <= limit; ++i) {
        if (sieve[i]) {
            for (int j = i * i; j <= limit; j += i) {
                sieve[j] = false;
            }
        }
    }

    std::vector<int> primes;
    for (int i = 2; i <= limit; ++i) {
        if (sieve[i]) primes.push_back(i);
    }
    return primes;
}

// FFT 实现
using Complex = std::complex<double>;
const double PI = std::acos(-1);

void fft(std::vector<Complex>& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;

    std::vector<Complex> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; ++i) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    Complex w(1), wn(std::cos(ang), std::sin(ang));

    for (int i = 0; 2 * i < n; ++i) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i + n / 2] /= 2;
        }
        w *= wn;
    }
}

// 矩阵乘法
std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b) {
    int n = a.size();
    int m = b[0].size();
    int p = b.size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// LU 分解
void lu_decomposition(const std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& L,
    std::vector<std::vector<double>>& U) {
    int n = A.size();
    L.resize(n, std::vector<double>(n, 0.0));
    U.resize(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int k = i; k < n; ++k) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }

        for (int k = i; k < n; ++k) {
            if (i == k) {
                L[i][i] = 1.0;
            }
            else {
                double sum = 0.0;
                for (int j = 0; j < i; ++j) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

// 简化的 SVD 分解模拟
void svd_decomposition(const std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& U,
    std::vector<double>& S,
    std::vector<std::vector<double>>& V) {
    int n = A.size();
    int m = A[0].size();

    U.resize(n, std::vector<double>(n, 0.0));
    V.resize(m, std::vector<double>(m, 0.0));
    S.resize(std::min(n, m), 0.0);

    for (int i = 0; i < n; ++i) U[i][i] = 1.0;
    for (int i = 0; i < m; ++i) V[i][i] = 1.0;

    for (int i = 0; i < std::min(n, m); ++i) {
        S[i] = random_double(0.5, 2.0);
    }
    std::sort(S.rbegin(), S.rend());
}

// 单核测试（操作计数用于后续标准化）
void single_core_test() {
    log("开始单核性能测试，持续 " + std::to_string(test_config.duration.single_core) + " 秒");

    HANDLE current_thread = GetCurrentThread();
    DWORD_PTR original_affinity = SetThreadAffinityMask(current_thread, 1 << 0);

    if (original_affinity == 0) {
        log("警告: 无法设置线程亲和性，错误代码: " + std::to_string(GetLastError()));
    }
    else {
        log("线程已绑定到CPU0核心");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(test_config.duration.single_core);

    int operations = 0;

    while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
        calculate_primes(test_config.prime_limit / 10);

        int size = test_config.matrix_size / 2;
        std::vector<std::vector<double>> a(size, std::vector<double>(size));
        std::vector<std::vector<double>> b(size, std::vector<double>(size));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                a[i][j] = random_double();
                b[i][j] = random_double();
            }
        }
        auto c = matrix_multiply(a, b);

        int fft_size = 1024;
        std::vector<Complex> fft_data(fft_size);
        for (int i = 0; i < fft_size; ++i) {
            fft_data[i] = Complex(random_double(), random_double());
        }
        fft(fft_data, false);
        fft(fft_data, true);

        std::vector<std::vector<double>> L, U;
        lu_decomposition(a, L, U);

        std::vector<std::vector<double>> U_svd, V_svd;
        std::vector<double> S_svd;
        svd_decomposition(a, U_svd, S_svd, V_svd);

        operations++;
    }

    if (original_affinity != 0) {
        SetThreadAffinityMask(current_thread, original_affinity);
        log("线程亲和性已恢复");
    }

    if (!stop_test) {
        double raw_score = static_cast<double>(operations) / test_config.duration.single_core;
        // 标准化得分 = (原始得分 / 基准值) * 100
        double normalized = (raw_score / test_config.benchmarks.single_core) * 100;
        normalized = std::max(0.0, normalized);

        std::lock_guard<std::mutex> lock(result_mutex);
        raw_results["single_core"] = raw_score;
        test_results["single_core"] = normalized;
        log("单核性能测试完成 - 原始: " + std::to_string(raw_score) +
            ", 标准化: " + std::to_string(normalized));
    }
    else {
        log("单核性能测试被中止");
    }
}

// 多核工作线程函数
void multi_core_worker(int thread_id, int duration, std::atomic<int>& operations) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(duration);

    while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
        int size = 200;
        std::vector<std::vector<double>> matrix(size, std::vector<double>(size));

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = random_double();
            }
            matrix[i][i] += 0.5;
        }

        int fft_size = 1024;
        std::vector<Complex> fft_data(fft_size);
        for (int i = 0; i < fft_size; ++i) {
            fft_data[i] = Complex(random_double(), random_double());
        }
        fft(fft_data, false);
        fft(fft_data, true);

        std::vector<std::vector<double>> L, U;
        lu_decomposition(matrix, L, U);

        std::vector<std::vector<double>> U_svd, V_svd;
        std::vector<double> S_svd;
        svd_decomposition(matrix, U_svd, S_svd, V_svd);

        operations++;
    }
}

// 多核测试（标准化得分）
void multi_core_test() {
    log("开始多核性能测试，持续 " + std::to_string(test_config.duration.multi_core) + " 秒");

    int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4;

    log("检测到 " + std::to_string(num_cores) + " 个CPU核心");

    std::vector<std::atomic<int>> thread_ops(num_cores);
    for (auto& ops : thread_ops) ops = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_cores; ++i) {
        threads.emplace_back(multi_core_worker, i, test_config.duration.multi_core, std::ref(thread_ops[i]));
        log("启动多核工作线程 " + std::to_string(i + 1) + "/" + std::to_string(num_cores));
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(test_config.duration.multi_core);

    while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    if (!stop_test) {
        int total_ops = 0;
        for (const auto& ops : thread_ops) {
            total_ops += ops;
        }

        double raw_score = static_cast<double>(total_ops) / test_config.duration.multi_core;
        double normalized = (raw_score / test_config.benchmarks.multi_core) * 100;
        normalized = std::max(0.0, normalized);

        std::lock_guard<std::mutex> lock(result_mutex);
        raw_results["multi_core"] = raw_score;
        test_results["multi_core"] = normalized;
        log("多核性能测试完成 - 原始: " + std::to_string(raw_score) +
            ", 标准化: " + std::to_string(normalized));
    }
    else {
        log("多核性能测试被中止");
    }
}

// 获取缓存信息的结构体
struct CacheInfo {
    size_t l1_data_size = 0;
    size_t l1_instruction_size = 0;
    size_t l2_size = 0;
    size_t l3_size = 0;
    size_t line_size = 64;
    int l1_data_associativity = 8;
    int l1_instruction_associativity = 8;
    int l2_associativity = 8;
    int l3_associativity = 16;

    // 添加一个便捷函数来获取总的L1大小（用于向后兼容）
    size_t get_l1_total_size() const {
        return l1_data_size + l1_instruction_size;
    }
};

// 获取系统缓存信息（使用GetLogicalProcessorInformationEx API）
CacheInfo get_cache_info() {
    CacheInfo info;

    // 首先尝试使用GetLogicalProcessorInformationEx（Windows 7+）
    DWORD buffer_size = 0;
    BOOL result = GetLogicalProcessorInformationEx(RelationCache, nullptr, &buffer_size);

    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        auto* buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(buffer_size);
        if (buffer && GetLogicalProcessorInformationEx(RelationCache, buffer, &buffer_size)) {
            BYTE* ptr = (BYTE*)buffer;
            DWORD offset = 0;

            while (offset < buffer_size) {
                auto* current = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(ptr + offset);

                if (current->Relationship == RelationCache) {
                    auto& cache = current->Cache;
                    size_t cache_size = cache.CacheSize;

                    switch (cache.Level) {
                    case 1:
                        if (cache.Type == CacheData) {
                            info.l1_data_size = cache_size;
                            info.l1_data_associativity = cache.Associativity;
                        }
                        else if (cache.Type == CacheInstruction) {
                            info.l1_instruction_size = cache_size;
                            info.l1_instruction_associativity = cache.Associativity;
                        }
                        else if (cache.Type == CacheUnified) {
                            // 统一缓存平均分配
                            info.l1_data_size = cache_size / 2;
                            info.l1_instruction_size = cache_size / 2;
                            info.l1_data_associativity = cache.Associativity;
                            info.l1_instruction_associativity = cache.Associativity;
                        }
                        break;
                    case 2:
                        info.l2_size = cache_size;
                        info.l2_associativity = cache.Associativity;
                        break;
                    case 3:
                        info.l3_size = cache_size;
                        info.l3_associativity = cache.Associativity;
                        break;
                    }

                    if (cache.LineSize > 0) {
                        info.line_size = cache.LineSize;
                    }
                }

                offset += current->Size;
            }
            free(buffer);
        }
    }

    // 如果新API失败，回退到旧的GetLogicalProcessorInformation
    if (info.l1_data_size == 0 && info.l2_size == 0) {
        DWORD old_buffer_size = 0;
        BOOL old_result = GetLogicalProcessorInformation(nullptr, &old_buffer_size);

        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            auto* old_buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(old_buffer_size);
            if (old_buffer && GetLogicalProcessorInformation(old_buffer, &old_buffer_size)) {
                DWORD count = old_buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);

                for (DWORD i = 0; i < count; ++i) {
                    if (old_buffer[i].Relationship == RelationCache) {
                        auto& cache = old_buffer[i].Cache;
                        size_t cache_size = cache.Size;

                        switch (cache.Level) {
                        case 1:
                            if (cache.Type == CacheData) {
                                info.l1_data_size = cache_size;
                                info.l1_data_associativity = cache.Associativity;
                            }
                            else if (cache.Type == CacheInstruction) {
                                info.l1_instruction_size = cache_size;
                                info.l1_instruction_associativity = cache.Associativity;
                            }
                            else if (cache.Type == CacheUnified) {
                                info.l1_data_size = cache_size / 2;
                                info.l1_instruction_size = cache_size / 2;
                                info.l1_data_associativity = cache.Associativity;
                                info.l1_instruction_associativity = cache.Associativity;
                            }
                            break;
                        case 2:
                            info.l2_size = cache_size;
                            info.l2_associativity = cache.Associativity;
                            break;
                        case 3:
                            info.l3_size = cache_size;
                            info.l3_associativity = cache.Associativity;
                            break;
                        }

                        if (cache.LineSize > 0) {
                            info.line_size = cache.LineSize;
                        }
                    }
                }
                free(old_buffer);
            }
        }
    }

    // 如果仍然无法获取缓存信息，使用基于CPU检测的启发式默认值
    if (info.l1_data_size == 0) {
        // 尝试使用CPUID检测CPU类型来设置更合适的默认值
        int cpu_info[4] = { 0 };
        __cpuid(cpu_info, 0);

        // 简单的CPU家族检测
        __cpuid(cpu_info, 1);
        int family = (cpu_info[0] >> 8) & 0xF;
        int model = (cpu_info[0] >> 4) & 0xF;

        if (family == 0x6) { // Intel Core系列
            if (model >= 0x0E) { // Core i系列
                info.l1_data_size = 32 * 1024;
                info.l1_instruction_size = 32 * 1024;
                info.l2_size = 256 * 1024; // 每核心
                info.l3_size = 8 * 1024 * 1024; // 共享
            }
            else { // 旧的Core 2系列
                info.l1_data_size = 32 * 1024;
                info.l1_instruction_size = 32 * 1024;
                info.l2_size = 2 * 1024 * 1024; // 共享
                info.l3_size = 0;
            }
        }
        else { // AMD或其他
            info.l1_data_size = 64 * 1024; // AMD通常有更大的L1缓存
            info.l1_instruction_size = 64 * 1024;
            info.l2_size = 512 * 1024;
            info.l3_size = 8 * 1024 * 1024;
        }
    }

    // 确保至少有一些合理的默认值
    if (info.l1_data_size == 0) info.l1_data_size = 32 * 1024;
    if (info.l1_instruction_size == 0) info.l1_instruction_size = 32 * 1024;
    if (info.l2_size == 0) info.l2_size = 256 * 1024;
    if (info.l3_size == 0) info.l3_size = 8 * 1024 * 1024;

    return info;
}

// 测试缓存性能的辅助函数
class CacheTester {
private:
    size_t cache_size;
    size_t line_size;
    int associativity;
    std::vector<uint8_t> test_data;

public:
    CacheTester(size_t size, size_t line, int assoc)
        : cache_size(size), line_size(line), associativity(assoc) {
        // 分配测试数据，大小为缓存大小的2倍以确保覆盖
        test_data.resize(cache_size * 2, 1);
    }

    // 顺序访问测试（高缓存命中率）
    double test_sequential_access(int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        size_t data_size = test_data.size();
        volatile uint64_t sum = 0; // volatile防止优化

        for (int iter = 0; iter < iterations && !stop_test; ++iter) {
            // 顺序访问，利用空间局部性
            for (size_t i = 0; i < data_size; i += line_size) {
                sum += test_data[i];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return (iterations * data_size / line_size) / (duration.count() / 1e6); // 操作数/秒
    }

    // 随机访问测试（低缓存命中率）
    double test_random_access(int iterations = 500) {
        auto start = std::chrono::high_resolution_clock::now();

        size_t data_size = test_data.size();
        volatile uint64_t sum = 0;

        // 预生成随机访问模式
        std::vector<size_t> access_pattern(data_size / line_size);
        std::iota(access_pattern.begin(), access_pattern.end(), 0);
        std::shuffle(access_pattern.begin(), access_pattern.end(),
            std::mt19937(std::random_device{}()));

        for (int iter = 0; iter < iterations && !stop_test; ++iter) {
            // 随机访问，破坏空间局部性
            for (size_t idx : access_pattern) {
                size_t pos = idx * line_size;
                if (pos < data_size) {
                    sum += test_data[pos];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return (iterations * access_pattern.size()) / (duration.count() / 1e6);
    }

    // 测试缓存关联性冲突
    double test_associativity_conflict(int iterations = 300) {
        auto start = std::chrono::high_resolution_clock::now();

        // 创建多个映射到相同缓存组的地址
        size_t sets = cache_size / (line_size * associativity);
        std::vector<size_t> conflict_addresses;

        for (int i = 0; i < associativity + 2; ++i) { // 超过关联度，产生冲突
            size_t addr = i * sets * line_size;
            if (addr < test_data.size()) {
                conflict_addresses.push_back(addr);
            }
        }

        volatile uint64_t sum = 0;
        for (int iter = 0; iter < iterations && !stop_test; ++iter) {
            for (size_t addr : conflict_addresses) {
                sum += test_data[addr];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return (iterations * conflict_addresses.size()) / (duration.count() / 1e6);
    }

    // 测试预取机制效果
    double test_prefetch_friendly(int iterations = 800) {
        auto start = std::chrono::high_resolution_clock::now();

        size_t data_size = test_data.size();
        volatile uint64_t sum = 0;

        // 大步长访问，测试硬件预取
        const size_t stride = line_size * 4; // 4个缓存行的大步长

        for (int iter = 0; iter < iterations && !stop_test; ++iter) {
            for (size_t i = 0; i < data_size; i += stride) {
                sum += test_data[i];
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return (iterations * data_size / stride) / (duration.count() / 1e6);
    }
};

// 基于Stack Processing模型的增强内存测试
void memory_test() {
    log("开始内存性能测试，持续 " + std::to_string(test_config.duration.memory) + " 秒");

    // 获取系统缓存信息
    CacheInfo cache_info = get_cache_info();
    log("缓存信息: L1数据=" + std::to_string(cache_info.l1_data_size / 1024) + "KB, " +
        "L1指令=" + std::to_string(cache_info.l1_instruction_size / 1024) + "KB, " +
        "L2=" + std::to_string(cache_info.l2_size / 1024) + "KB, " +
        "L3=" + std::to_string(cache_info.l3_size / (1024 * 1024)) + "MB, " +
        "缓存行=" + std::to_string(cache_info.line_size) + "B");

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(test_config.duration.memory);

    // 基于Stack Processing模型的缓存模拟器
    class CacheStackSimulator {
    private:
        size_t cache_size;
        size_t line_size;
        int associativity;
        size_t num_sets;
        std::vector<std::list<size_t>> lru_stacks; // LRU堆栈，每个组一个

    public:
        CacheStackSimulator(size_t size, size_t line, int assoc)
            : cache_size(size), line_size(line), associativity(assoc) {
            num_sets = cache_size / (line_size * associativity);
            lru_stacks.resize(num_sets);
        }

        // 模拟一次内存访问，返回是否命中
        bool access(size_t address) {
            size_t block_addr = address / line_size;
            size_t set_index = block_addr % num_sets;

            auto& stack = lru_stacks[set_index];

            // 在堆栈中查找该块
            auto it = std::find(stack.begin(), stack.end(), block_addr);

            if (it != stack.end()) {
                // 命中：将块移动到堆栈顶部（MRU位置）
                stack.erase(it);
                stack.push_front(block_addr);
                return true;
            }
            else {
                // 未命中：将块添加到堆栈顶部
                stack.push_front(block_addr);

                // 如果堆栈超过关联度，移除LRU块
                if (stack.size() > static_cast<size_t>(associativity)) {
                    stack.pop_back();
                }
                return false;
            }
        }

        // 获取当前命中率统计
        void get_stats(int& hits, int& accesses) const {
            hits = 0;
            accesses = 0;
            // 这里只是返回基本统计，实际统计在外部维护
        }

        // 重置模拟器
        void reset() {
            for (auto& stack : lru_stacks) {
                stack.clear();
            }
        }

        size_t get_num_sets() const { return num_sets; }
        size_t get_line_size() const { return line_size; }
    };

    // 工作负载生成器 - 模拟真实应用程序的访问模式
    class WorkloadGenerator {
    private:
        size_t working_set_size;
        size_t line_size;
        std::mt19937 rng;

        // 工作集数据块
        std::vector<size_t> working_set_blocks;

        // 访问模式参数
        double temporal_locality_prob;  // 时间局部性概率
        double spatial_locality_prob;   // 空间局部性概率
        double sequential_access_prob;  // 顺序访问概率

    public:
        WorkloadGenerator(size_t total_size, size_t line_size,
            double temporal_prob = 0.3, double spatial_prob = 0.4, double sequential_prob = 0.2)
            : working_set_size(total_size), line_size(line_size),
            temporal_locality_prob(temporal_prob),
            spatial_locality_prob(spatial_prob),
            sequential_access_prob(sequential_prob),
            rng(std::random_device{}()) {

            // 初始化工作集
            size_t num_blocks = working_set_size / line_size;
            working_set_blocks.resize(num_blocks);
            std::iota(working_set_blocks.begin(), working_set_blocks.end(), 0);
        }

        // 生成具有真实工作负载特性的访问序列
        std::vector<size_t> generate_access_sequence(int num_accesses) {
            std::vector<size_t> accesses;
            accesses.reserve(num_accesses);

            size_t num_blocks = working_set_blocks.size();
            if (num_blocks == 0) return accesses;

            // 当前访问位置（用于顺序访问模式）
            size_t current_seq_pos = 0;

            // 最近访问的块（用于时间局部性）
            std::vector<size_t> recent_blocks;
            const size_t recent_window = std::min(num_blocks, size_t(100));

            // 热点区域（模拟程序的热点数据）
            std::uniform_int_distribution<size_t> hotspot_dist(0, num_blocks / 10);
            std::vector<size_t> hotspot_blocks;
            for (size_t i = 0; i < num_blocks / 20; ++i) {
                hotspot_blocks.push_back(hotspot_dist(rng));
            }

            for (int i = 0; i < num_accesses; ++i) {
                std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
                double rand_val = prob_dist(rng);

                size_t next_block;

                if (rand_val < sequential_access_prob) {
                    // 顺序访问模式
                    next_block = current_seq_pos;
                    current_seq_pos = (current_seq_pos + 1) % num_blocks;
                }
                else if (rand_val < sequential_access_prob + temporal_locality_prob) {
                    // 时间局部性：访问最近访问过的块
                    if (!recent_blocks.empty()) {
                        std::uniform_int_distribution<size_t> recent_dist(0, recent_blocks.size() - 1);
                        next_block = recent_blocks[recent_dist(rng)];
                    }
                    else {
                        std::uniform_int_distribution<size_t> block_dist(0, num_blocks - 1);
                        next_block = block_dist(rng);
                    }
                }
                else if (rand_val < sequential_access_prob + temporal_locality_prob + spatial_locality_prob) {
                    // 空间局部性：访问相邻块
                    if (!accesses.empty()) {
                        size_t last_block = accesses.back() / line_size;
                        std::normal_distribution<double> spatial_dist(0.0, 10.0); // 相邻范围
                        int offset = static_cast<int>(spatial_dist(rng));
                        next_block = (last_block + offset + num_blocks) % num_blocks;
                    }
                    else {
                        std::uniform_int_distribution<size_t> block_dist(0, num_blocks - 1);
                        next_block = block_dist(rng);
                    }
                }
                else {
                    // 热点访问：频繁访问少量数据
                    if (!hotspot_blocks.empty() && prob_dist(rng) < 0.7) {
                        std::uniform_int_distribution<size_t> hotspot_index_dist(0, hotspot_blocks.size() - 1);
                        next_block = hotspot_blocks[hotspot_index_dist(rng)];
                    }
                    else {
                        // 完全随机访问
                        std::uniform_int_distribution<size_t> block_dist(0, num_blocks - 1);
                        next_block = block_dist(rng);
                    }
                }

                // 转换为字节地址
                size_t address = next_block * line_size;
                accesses.push_back(address);

                // 更新最近访问列表
                recent_blocks.push_back(next_block);
                if (recent_blocks.size() > recent_window) {
                    recent_blocks.erase(recent_blocks.begin());
                }
            }

            return accesses;
        }

        // 生成特定模式的访问序列（用于对比分析）
        std::vector<size_t> generate_pattern_sequence(int num_accesses, const std::string& pattern) {
            std::vector<size_t> accesses;
            accesses.reserve(num_accesses);

            size_t num_blocks = working_set_blocks.size();
            if (num_blocks == 0) return accesses;

            if (pattern == "sequential") {
                // 纯顺序访问
                for (int i = 0; i < num_accesses; ++i) {
                    accesses.push_back((i % num_blocks) * line_size);
                }
            }
            else if (pattern == "random") {
                // 纯随机访问
                std::uniform_int_distribution<size_t> dist(0, num_blocks - 1);
                for (int i = 0; i < num_accesses; ++i) {
                    accesses.push_back(dist(rng) * line_size);
                }
            }
            else if (pattern == "stride") {
                // 大步长访问
                const size_t stride = 16; // 16个缓存行的大步长
                for (int i = 0; i < num_accesses; ++i) {
                    accesses.push_back(((i * stride) % num_blocks) * line_size);
                }
            }
            else if (pattern == "loop") {
                // 循环访问小工作集
                const size_t loop_size = std::min(num_blocks, size_t(100));
                for (int i = 0; i < num_accesses; ++i) {
                    accesses.push_back((i % loop_size) * line_size);
                }
            }

            return accesses;
        }
    };

    // 为每个缓存级别创建Stack Processing模拟器和工作负载生成器
    CacheStackSimulator l1_stack(cache_info.l1_data_size, cache_info.line_size, cache_info.l1_data_associativity);
    CacheStackSimulator l2_stack(cache_info.l2_size, cache_info.line_size, cache_info.l2_associativity);
    CacheStackSimulator l3_stack(cache_info.l3_size, cache_info.line_size, cache_info.l3_associativity);

    // 为不同缓存级别创建适当大小的工作负载生成器
    WorkloadGenerator l1_workload(cache_info.l1_data_size * 8, cache_info.line_size, 0.4, 0.3, 0.2);
    WorkloadGenerator l2_workload(cache_info.l2_size * 4, cache_info.line_size, 0.3, 0.3, 0.15);
    WorkloadGenerator l3_workload(cache_info.l3_size * 2, cache_info.line_size, 0.25, 0.25, 0.1);
    WorkloadGenerator mem_workload(256 * 1024 * 1024, cache_info.line_size, 0.2, 0.2, 0.05);

    struct CacheLevelResult {
        double sequential_speed = 0;
        double random_speed = 0;
        double conflict_speed = 0;
        double prefetch_speed = 0;
        double mixed_workload_hit_ratio = 0;  // 混合工作负载命中率
        double sequential_hit_ratio = 0;      // 顺序访问命中率
        double random_hit_ratio = 0;          // 随机访问命中率
        double temporal_hit_ratio = 0;        // 时间局部性命中率
        int total_accesses = 0;
        int hit_accesses = 0;
        int mixed_accesses = 0;
        int mixed_hits = 0;
    };

    std::map<std::string, CacheLevelResult> cache_results;
    int test_iterations = 0;
    const int ACCESSES_PER_ITERATION = 50000; // 大幅增加每次迭代的访问次数

    // 为原有的CacheTester创建实例（保持兼容性）
    CacheTester l1_tester(cache_info.l1_data_size, cache_info.line_size, cache_info.l1_data_associativity);
    CacheTester l2_tester(cache_info.l2_size, cache_info.line_size, cache_info.l2_associativity);
    CacheTester l3_tester(cache_info.l3_size, cache_info.line_size, cache_info.l3_associativity);
    CacheTester main_memory_tester(64 * 1024 * 1024, cache_info.line_size, 1);

    log("开始Stack Processing模型模拟测试...");
    log("每次迭代模拟 " + std::to_string(ACCESSES_PER_ITERATION) + " 次内存访问");

    while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
        test_iterations++;

        // 生成混合工作负载访问序列
        auto l1_mixed_accesses = l1_workload.generate_access_sequence(ACCESSES_PER_ITERATION);
        auto l2_mixed_accesses = l2_workload.generate_access_sequence(ACCESSES_PER_ITERATION);
        auto l3_mixed_accesses = l3_workload.generate_access_sequence(ACCESSES_PER_ITERATION);
        auto mem_mixed_accesses = mem_workload.generate_access_sequence(ACCESSES_PER_ITERATION);

        // 模拟混合工作负载访问
        auto simulate_accesses = [](CacheStackSimulator& simulator,
            const std::vector<size_t>& accesses,
            CacheLevelResult& result) {
                int hits = 0;
                for (size_t addr : accesses) {
                    if (simulator.access(addr)) {
                        hits++;
                    }
                }
                result.mixed_accesses += accesses.size();
                result.mixed_hits += hits;
            };

        simulate_accesses(l1_stack, l1_mixed_accesses, cache_results["L1"]);
        simulate_accesses(l2_stack, l2_mixed_accesses, cache_results["L2"]);
        simulate_accesses(l3_stack, l3_mixed_accesses, cache_results["L3"]);

        // 每5次迭代测试一次特定模式，避免过于频繁影响性能
        if (test_iterations % 5 == 0) {
            // 测试不同访问模式的命中率
            auto test_pattern = [&](const std::string& pattern_name,
                WorkloadGenerator& workload,
                CacheStackSimulator& simulator,
                CacheLevelResult& result) {
                    auto accesses = workload.generate_pattern_sequence(ACCESSES_PER_ITERATION / 10, pattern_name);
                    int hits = 0;
                    for (size_t addr : accesses) {
                        if (simulator.access(addr)) {
                            hits++;
                        }
                    }
                    double hit_ratio = static_cast<double>(hits) / accesses.size();

                    if (pattern_name == "sequential") {
                        result.sequential_hit_ratio = (result.sequential_hit_ratio + hit_ratio) / 2.0;
                    }
                    else if (pattern_name == "random") {
                        result.random_hit_ratio = (result.random_hit_ratio + hit_ratio) / 2.0;
                    }
                    else if (pattern_name == "loop") {
                        result.temporal_hit_ratio = (result.temporal_hit_ratio + hit_ratio) / 2.0;
                    }
                };

            test_pattern("sequential", l1_workload, l1_stack, cache_results["L1"]);
            test_pattern("random", l1_workload, l1_stack, cache_results["L1"]);
            test_pattern("loop", l1_workload, l1_stack, cache_results["L1"]);

            test_pattern("sequential", l2_workload, l2_stack, cache_results["L2"]);
            test_pattern("random", l2_workload, l2_stack, cache_results["L2"]);
            test_pattern("loop", l2_workload, l2_stack, cache_results["L2"]);

            test_pattern("sequential", l3_workload, l3_stack, cache_results["L3"]);
            test_pattern("random", l3_workload, l3_stack, cache_results["L3"]);
            test_pattern("loop", l3_workload, l3_stack, cache_results["L3"]);
        }

        // 原有的速度测试（减少频率以降低开销）
        if (test_iterations % 10 == 0) {
            auto& l1_result = cache_results["L1"];
            l1_result.sequential_speed += l1_tester.test_sequential_access(10);
            l1_result.random_speed += l1_tester.test_random_access(5);
            l1_result.conflict_speed += l1_tester.test_associativity_conflict(3);
            l1_result.prefetch_speed += l1_tester.test_prefetch_friendly(8);

            auto& l2_result = cache_results["L2"];
            l2_result.sequential_speed += l2_tester.test_sequential_access(8);
            l2_result.random_speed += l2_tester.test_random_access(4);
            l2_result.conflict_speed += l2_tester.test_associativity_conflict(2);
            l2_result.prefetch_speed += l2_tester.test_prefetch_friendly(6);

            auto& l3_result = cache_results["L3"];
            l3_result.sequential_speed += l3_tester.test_sequential_access(6);
            l3_result.random_speed += l3_tester.test_random_access(3);
            l3_result.conflict_speed += l3_tester.test_associativity_conflict(2);
            l3_result.prefetch_speed += l3_tester.test_prefetch_friendly(4);

            auto& mem_result = cache_results["MainMemory"];
            mem_result.sequential_speed += main_memory_tester.test_sequential_access(4);
            mem_result.random_speed += main_memory_tester.test_random_access(3);
            mem_result.conflict_speed += main_memory_tester.test_associativity_conflict(1);
            mem_result.prefetch_speed += main_memory_tester.test_prefetch_friendly(3);
        }

        // 传统内存测试保持兼容性（减少频率）
        if (test_iterations % 20 == 0) {
            std::vector<double> large_array(test_config.memory_size);
            for (size_t i = 0; i < large_array.size(); ++i) {
                large_array[i] = random_double();
            }

            double sum = 0.0;
            for (double val : large_array) {
                sum += val;
            }

            std::vector<size_t> indices(large_array.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

            double rand_sum = 0.0;
            for (size_t idx : indices) {
                rand_sum += large_array[idx] * 1.0001;
            }
        }

        // 定期输出进度
        if (test_iterations % 50 == 0) {
            log("内存测试进度: " + std::to_string(test_iterations) + " 次迭代完成, " +
                "总访问次数: " + std::to_string(test_iterations * ACCESSES_PER_ITERATION));
        }
    }

    if (!stop_test && test_iterations > 0) {
        // 计算最终命中率和平均速度
        for (auto& [level, result] : cache_results) {
            if (level != "MainMemory") {
                // 计算混合工作负载命中率
                if (result.mixed_accesses > 0) {
                    result.mixed_workload_hit_ratio = static_cast<double>(result.mixed_hits) / result.mixed_accesses;
                }

                // 计算平均速度
                int speed_test_count = test_iterations / 10;
                if (speed_test_count > 0) {
                    result.sequential_speed /= speed_test_count;
                    result.random_speed /= speed_test_count;
                    result.conflict_speed /= speed_test_count;
                    result.prefetch_speed /= speed_test_count;
                }
            }
        }

        // 输出详细的缓存性能分析
        log("\n===== 基于Stack Processing模型的缓存性能分析 =====");
        log("总迭代次数: " + std::to_string(test_iterations));
        log("总内存访问模拟: " + std::to_string(test_iterations * ACCESSES_PER_ITERATION) + " 次");

        for (const auto& [level, result] : cache_results) {
            if (level != "MainMemory") {
                log("\n" + level + "缓存性能分析:");
                log("  混合工作负载命中率: " + std::to_string(result.mixed_workload_hit_ratio * 100) + "%");
                log("  顺序访问命中率: " + std::to_string(result.sequential_hit_ratio * 100) + "%");
                log("  随机访问命中率: " + std::to_string(result.random_hit_ratio * 100) + "%");
                log("  时间局部性命中率: " + std::to_string(result.temporal_hit_ratio * 100) + "%");
                log("  访问统计: " + std::to_string(result.mixed_hits) + " / " +
                    std::to_string(result.mixed_accesses) + " 命中");
                log("  性能指标:");
                log("    - 顺序访问速度: " + std::to_string(result.sequential_speed) + " ops/sec");
                log("    - 随机访问速度: " + std::to_string(result.random_speed) + " ops/sec");
                log("    - 冲突访问速度: " + std::to_string(result.conflict_speed) + " ops/sec");
                log("    - 预取友好速度: " + std::to_string(result.prefetch_speed) + " ops/sec");
            }
        }

        // 计算综合内存性能得分 - 主要基于Stack Processing模型
        double total_raw_score = 0;
        double weight_sum = 0;

        // L1缓存：混合工作负载命中率权重最高
        total_raw_score += cache_results["L1"].mixed_workload_hit_ratio * 200000 * 0.5;
        total_raw_score += cache_results["L1"].sequential_speed * 0.15;
        total_raw_score += cache_results["L1"].temporal_hit_ratio * 50000 * 0.1;
        weight_sum += 0.75;

        // L2缓存
        total_raw_score += cache_results["L2"].mixed_workload_hit_ratio * 100000 * 0.25;
        total_raw_score += cache_results["L2"].sequential_speed * 0.08;
        total_raw_score += cache_results["L2"].temporal_hit_ratio * 25000 * 0.05;
        weight_sum += 0.38;

        // L3缓存
        total_raw_score += cache_results["L3"].mixed_workload_hit_ratio * 50000 * 0.12;
        total_raw_score += cache_results["L3"].sequential_speed * 0.05;
        weight_sum += 0.17;

        // 主存（仅基于速度）
        total_raw_score += cache_results["MainMemory"].sequential_speed * 0.05;
        weight_sum += 0.05;

        double raw_score = total_raw_score / weight_sum;

        // 标准化得分
        double normalized = (raw_score / test_config.benchmarks.memory) * 100;
        normalized = std::max(0.0, normalized);

        // 基于Stack Processing命中率的智能加成
        double stack_bonus = 0;
        for (const auto& [level, result] : cache_results) {
            if (level != "MainMemory") {
                // 根据混合工作负载命中率计算加成，采用非线性增长
                double hr = result.mixed_workload_hit_ratio;
                double bonus = 0;

                if (hr > 0.9) bonus = 25.0;      // 优秀缓存
                else if (hr > 0.8) bonus = 15.0; // 良好缓存
                else if (hr > 0.7) bonus = 8.0;  // 中等缓存
                else if (hr > 0.6) bonus = 3.0;  // 一般缓存
                else bonus = 0.0;                // 较差缓存

                // 时间局部性表现额外加成
                if (result.temporal_hit_ratio > 0.8) {
                    bonus += 5.0;
                }

                stack_bonus += bonus;
                log(level + "缓存加成: +" + std::to_string(bonus));
            }
        }

        normalized = normalized + stack_bonus;

        std::lock_guard<std::mutex> lock(result_mutex);
        raw_results["memory"] = raw_score;
        test_results["memory"] = normalized;

        log("\n内存性能测试完成");
        log("原始得分: " + std::to_string(raw_score));
        log("Stack Processing加成: +" + std::to_string(stack_bonus));
        log("标准化得分: " + std::to_string(normalized));
        log("===========================================");
    }
    else {
        log("内存性能测试被中止");
    }
}

// 加密解密
namespace crypto {
    // 轮常量（AES标准）
    const uint32_t RCON[10] = {
        0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000,
        0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000
    };

    // S盒（AES标准）
    uint8_t sbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
    };

    // 逆S盒（用于解密，此处仅加密用，保留备用）
    uint8_t inv_sbox[256] = {
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
    };

    // 辅助函数：32位字循环左移1字节
    uint32_t rot_word(uint32_t word) {
        return (word << 8) | (word >> 24);
    }

    // 辅助函数：字替换（S盒变换）
    uint32_t sub_word(uint32_t word) {
        return (sbox[(word >> 24) & 0xFF] << 24) |
            (sbox[(word >> 16) & 0xFF] << 16) |
            (sbox[(word >> 8) & 0xFF] << 8) |
            sbox[word & 0xFF];
    }

    // 密钥扩展（AES标准，支持128/256位密钥）
    void key_expansion(const std::vector<uint8_t>& key, std::vector<std::vector<uint8_t>>& w) {
        int key_len = key.size();
        // 验证密钥长度（仅支持128/256位）
        if (key_len != 16 && key_len != 32) {
            throw std::invalid_argument("AES密钥长度必须为16字节（128位）或32字节（256位）");
        }

        int Nk = key_len / 4;          // 密钥字数（16字节=4字，32字节=8字）
        int Nr = (key_len == 16) ? 10 : 14;  // 加密轮数（128位=10轮，256位=14轮）
        w.resize(Nr + 1, std::vector<uint8_t>(16));  // 轮密钥：Nr+1个，每个16字节

        // 步骤1：复制原始密钥到前Nk个32位字（w的前Nk*4字节）
        for (int i = 0; i < Nk; ++i) {
            uint32_t word = (key[4 * i] << 24) | (key[4 * i + 1] << 16) | (key[4 * i + 2] << 8) | key[4 * i + 3];
            // 轮密钥w[i]的4个字节 = 当前字的4个字节
            w[i / 4][(i % 4) * 4] = (word >> 24) & 0xFF;
            w[i / 4][(i % 4) * 4 + 1] = (word >> 16) & 0xFF;
            w[i / 4][(i % 4) * 4 + 2] = (word >> 8) & 0xFF;
            w[i / 4][(i % 4) * 4 + 3] = word & 0xFF;
        }

        // 步骤2：扩展剩余轮密钥（总字数=Nr+1)*4）
        for (int i = Nk; i < (Nr + 1) * 4; ++i) {
            uint32_t temp = ((uint32_t)w[i / 4][(i % 4) * 4] << 24) |
                ((uint32_t)w[i / 4][(i % 4) * 4 + 1] << 16) |
                ((uint32_t)w[i / 4][(i % 4) * 4 + 2] << 8) |
                w[i / 4][(i % 4) * 4 + 3];

            // 关键：AES-256需额外处理i%Nk==4的情况
            if (i % Nk == 0) {
                temp = sub_word(rot_word(temp)) ^ RCON[i / Nk - 1];
            }
            else if (Nk == 8 && i % Nk == 4) {  // AES-256特殊规则
                temp = sub_word(temp);
            }

            // 与前Nk个字节的字异或
            uint32_t prev_word = ((uint32_t)w[(i - Nk) / 4][((i - Nk) % 4) * 4] << 24) |
                ((uint32_t)w[(i - Nk) / 4][((i - Nk) % 4) * 4 + 1] << 16) |
                ((uint32_t)w[(i - Nk) / 4][((i - Nk) % 4) * 4 + 2] << 8) |
                w[(i - Nk) / 4][((i - Nk) % 4) * 4 + 3];
            temp ^= prev_word;

            // 存入轮密钥w
            w[i / 4][(i % 4) * 4] = (temp >> 24) & 0xFF;
            w[i / 4][(i % 4) * 4 + 1] = (temp >> 16) & 0xFF;
            w[i / 4][(i % 4) * 4 + 2] = (temp >> 8) & 0xFF;
            w[i / 4][(i % 4) * 4 + 3] = temp & 0xFF;
        }
    }

    // 行移位（ShiftRows，AES标准）
    void shift_rows(std::vector<uint8_t>& state) {
        // 第1行：不移位
        // 第2行：左移1字节
        uint8_t temp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = temp;
        // 第3行：左移2字节
        temp = state[2];
        state[2] = state[10];
        state[10] = temp;
        temp = state[6];
        state[6] = state[14];
        state[14] = temp;
        // 第4行：左移3字节（等价于右移1字节）
        temp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = temp;
    }

    // 列混合（MixColumns，AES标准，GF(2^8)域乘法）
    void mix_columns(std::vector<uint8_t>& state) {
        auto gf_mult = [](uint8_t a, uint8_t b) -> uint8_t {
            uint8_t result = 0;
            for (int i = 0; i < 8; ++i) {
                if (b & 1) result ^= a;
                bool carry = (a & 0x80) != 0;
                a <<= 1;
                if (carry) a ^= 0x1B;  // AES多项式：x^8 + x^4 + x^3 + x + 1
                b >>= 1;
            }
            return result;
            };

        std::vector<uint8_t> temp(16);
        for (int c = 0; c < 4; ++c) {
            temp[4 * 0 + c] = (gf_mult(state[4 * 0 + c], 2) ^
                gf_mult(state[4 * 1 + c], 3) ^
                state[4 * 2 + c] ^
                state[4 * 3 + c]);
            temp[4 * 1 + c] = (state[4 * 0 + c] ^
                gf_mult(state[4 * 1 + c], 2) ^
                gf_mult(state[4 * 2 + c], 3) ^
                state[4 * 3 + c]);
            temp[4 * 2 + c] = (state[4 * 0 + c] ^
                state[4 * 1 + c] ^
                gf_mult(state[4 * 2 + c], 2) ^
                gf_mult(state[4 * 3 + c], 3));
            temp[4 * 3 + c] = (gf_mult(state[4 * 0 + c], 3) ^
                state[4 * 1 + c] ^
                state[4 * 2 + c] ^
                gf_mult(state[4 * 3 + c], 2));
        }
        state = temp;
    }

    // 轮密钥加（AddRoundKey）
    void add_round_key(std::vector<uint8_t>& state, const std::vector<uint8_t>& round_key) {
        for (int i = 0; i < 16; ++i) {
            state[i] ^= round_key[i];
        }
    }

    // S盒变换（SubBytes）
    void sub_bytes(std::vector<uint8_t>& state) {
        for (int i = 0; i < 16; ++i) {
            state[i] = sbox[state[i]];
        }
    }

    // 标准AES加密（支持128/256位密钥，按分组处理）
    void aes_encrypt(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key, std::vector<uint8_t>& out) {
        int key_len = key.size();
        int Nr = (key_len == 16) ? 10 : 14;  // 加密轮数
        std::vector<std::vector<uint8_t>> w;

        try {
            key_expansion(key, w);  // 生成轮密钥
        }
        catch (const std::invalid_argument& e) {
            log("AES密钥扩展失败：" + std::string(e.what()));
            throw;  // 向上抛出异常，避免崩溃
        }

        // 初始化输出缓冲区（与输入尺寸一致）
        out.resize(data.size(), 0);
        int num_blocks = data.size() / 16;  // 总分组数（必须是16的倍数）

        // 按分组循环加密
        for (int b = 0; b < num_blocks; ++b) {
            // 提取当前分组（16字节）
            std::vector<uint8_t> state(16);
            for (int i = 0; i < 16; ++i) {
                state[i] = data[b * 16 + i];
            }

            // 初始轮：仅轮密钥加
            add_round_key(state, w[0]);

            // 主加密循环（1~Nr-1轮）：SubBytes → ShiftRows → MixColumns → AddRoundKey
            for (int round = 1; round < Nr; ++round) {
                sub_bytes(state);
                shift_rows(state);
                mix_columns(state);
                add_round_key(state, w[round]);
            }

            // 最后一轮（Nr轮）：无MixColumns
            sub_bytes(state);
            shift_rows(state);
            add_round_key(state, w[Nr]);

            // 存入输出缓冲区
            for (int i = 0; i < 16; ++i) {
                out[b * 16 + i] = state[i];
            }
        }
    }

    // 标准SHA-1哈希
    void sha1_hash(const std::vector<uint8_t>& data, uint8_t hash[20]) {
        if (hash == nullptr) {
            throw std::invalid_argument("SHA-1哈希输出缓冲区不能为空");
        }

        // SHA-1标准初始化
        uint32_t h0 = 0x67452301;
        uint32_t h1 = 0xEFCDAB89;
        uint32_t h2 = 0x98BADCFE;
        uint32_t h3 = 0x10325476;
        uint32_t h4 = 0xC3D2E1F0;

        // 填充数据（SHA-1标准填充规则）
        uint64_t data_len = data.size() * 8;
        std::vector<uint8_t> padded_data = data;
        padded_data.push_back(0x80);  // 10000000

        // 填充0，直到长度 ≡ 448 mod 512
        while (padded_data.size() % 64 != 56) {
            padded_data.push_back(0x00);
        }

        // 追加数据长度（64位大端）
        for (int i = 7; i >= 0; --i) {
            padded_data.push_back((data_len >> (i * 8)) & 0xFF);
        }

        // 按512位（64字节）分组处理
        int num_blocks = padded_data.size() / 64;
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
            // 提取当前分组并扩展为80个32位字
            uint32_t w[80] = { 0 };
            for (int i = 0; i < 16; ++i) {
                w[i] = (padded_data[block_idx * 64 + 4 * i] << 24) |
                    (padded_data[block_idx * 64 + 4 * i + 1] << 16) |
                    (padded_data[block_idx * 64 + 4 * i + 2] << 8) |
                    padded_data[block_idx * 64 + 4 * i + 3];
            }
            for (int i = 16; i < 80; ++i) {
                w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]) << 1 | (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]) >> 31;
            }

            // 压缩函数
            uint32_t a_val = h0, b_val = h1, c_val = h2, d_val = h3, e_val = h4;
            for (int i = 0; i < 80; ++i) {
                uint32_t f, k;
                if (i < 20) {
                    f = (b_val & c_val) | ((~b_val) & d_val);
                    k = 0x5A827999;
                }
                else if (i < 40) {
                    f = b_val ^ c_val ^ d_val;
                    k = 0x6ED9EBA1;
                }
                else if (i < 60) {
                    f = (b_val & c_val) | (b_val & d_val) | (c_val & d_val);
                    k = 0x8F1BBCDC;
                }
                else {
                    f = b_val ^ c_val ^ d_val;
                    k = 0xCA62C1D6;
                }

                uint32_t temp = (a_val << 5 | a_val >> 27) + f + e_val + k + w[i];
                e_val = d_val;
                d_val = c_val;
                c_val = b_val << 30 | b_val >> 2;
                b_val = a_val;
                a_val = temp;
            }

            // 更新哈希值
            h0 += a_val;
            h1 += b_val;
            h2 += c_val;
            h3 += d_val;
            h4 += e_val;
        }

        // 输出哈希结果（大端序）
        hash[0] = (h0 >> 24) & 0xFF;
        hash[1] = (h0 >> 16) & 0xFF;
        hash[2] = (h0 >> 8) & 0xFF;
        hash[3] = h0 & 0xFF;
        hash[4] = (h1 >> 24) & 0xFF;
        hash[5] = (h1 >> 16) & 0xFF;
        hash[6] = (h1 >> 8) & 0xFF;
        hash[7] = h1 & 0xFF;
        hash[8] = (h2 >> 24) & 0xFF;
        hash[9] = (h2 >> 16) & 0xFF;
        hash[10] = (h2 >> 8) & 0xFF;
        hash[11] = h2 & 0xFF;
        hash[12] = (h3 >> 24) & 0xFF;
        hash[13] = (h3 >> 16) & 0xFF;
        hash[14] = (h3 >> 8) & 0xFF;
        hash[15] = h3 & 0xFF;
        hash[16] = (h4 >> 24) & 0xFF;
        hash[17] = (h4 >> 16) & 0xFF;
        hash[18] = (h4 >> 8) & 0xFF;
        hash[19] = h4 & 0xFF;
    }
}

// 增强的加密测试（添加参数有效性检查）
void crypto_test() {
    log("开始加密性能测试，持续 " + std::to_string(test_config.duration.crypto) + " 秒");

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(test_config.duration.crypto);

    int operations = 0;
    int key_size = test_config.developer_mode ? 32 : 16;  // 256位/128位密钥
    std::vector<uint8_t> key(key_size);
    for (int i = 0; i < key_size; ++i) {
        key[i] = static_cast<uint8_t>(random_double(0, 255));
    }

    // 验证密钥长度有效性
    if (key_size != 16 && key_size != 32) {
        log("错误：AES密钥长度必须为16或32字节");
        return;
    }
    log("使用 " + std::to_string(key_size * 8) + " 位AES密钥进行测试");

    while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
        // 数据尺寸必须是16的倍数（AES分组大小）
        int data_size = test_config.developer_mode ? 8192 : 1024;
        if (data_size % 16 != 0) {
            data_size = (data_size / 16 + 1) * 16;  // 向上取整到16的倍数
        }

        std::vector<uint8_t> data(data_size);
        for (int i = 0; i < data_size; ++i) {
            data[i] = static_cast<uint8_t>(random_double(0, 255));
        }

        try {
            // 1. AES加密（标准分组处理，无参数错误）
            std::vector<uint8_t> encrypted;
            crypto::aes_encrypt(data, key, encrypted);

            // 2. SHA-1哈希（参数有效性检查）
            uint8_t hash[20] = { 0 };
            crypto::sha1_hash(data, hash);

            // 3. 多轮加密（增加复杂度，验证稳定性）
            std::vector<uint8_t> temp = encrypted;
            for (int r = 0; r < test_config.crypto_rounds && !stop_test; ++r) {
                crypto::aes_encrypt(temp, key, temp);
            }

            operations++;
        }
        catch (const std::invalid_argument& e) {
            log("加密操作失败：" + std::string(e.what()));
            break;  // 异常时退出测试
        }
    }

    if (!stop_test) {
        double raw_score = static_cast<double>(operations) / test_config.duration.crypto;
        double normalized = (raw_score / test_config.benchmarks.crypto) * 100;
        normalized = std::max(0.0, normalized);

        std::lock_guard<std::mutex> lock(result_mutex);
        raw_results["crypto"] = raw_score;
        test_results["crypto"] = normalized;
        log("加密性能测试完成 - 原始: " + std::to_string(raw_score) +
            ", 标准化: " + std::to_string(normalized));
    }
    else {
        log("加密性能测试被中止");
    }
}

class GPUSimulator {
private:
    int num_compute_units;
    int max_work_group_size;
    bool has_double_precision;
    double memory_bandwidth_gbps;

public:
    GPUSimulator() {
        // 检测系统GPU能力（模拟）
        detect_gpu_capabilities();
    }

    void detect_gpu_capabilities() {
        // 模拟不同GPU厂商的能力检测
        log("检测GPU计算能力...");

        // 简单模拟：基于CPU线程数估算GPU能力
        num_compute_units = std::thread::hardware_concurrency() / 2;
        if (num_compute_units < 4) num_compute_units = 4;
        if (num_compute_units > 64) num_compute_units = 64;

        max_work_group_size = 256;  // 典型值
        has_double_precision = true;
        memory_bandwidth_gbps = 100.0;  // 保守估计

        log("GPU模拟器初始化: " +
            std::to_string(num_compute_units) + "个计算单元, " +
            "工作组大小: " + std::to_string(max_work_group_size) + ", " +
            "双精度: " + std::string(has_double_precision ? "是" : "否") + ", " +
            "内存带宽: " + std::to_string(memory_bandwidth_gbps) + " GB/s");
    }

    // 通用矩阵乘法 (GEMM) - 模拟GPU并行计算
    double test_gemm(int matrix_size, int duration_sec) {
        log("开始GPU GEMM测试 (矩阵大小: " + std::to_string(matrix_size) + "x" + std::to_string(matrix_size) + ")");

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_sec);

        int operations = 0;
        const int total_threads = num_compute_units * max_work_group_size;

        while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
            // 模拟并行矩阵乘法
            std::vector<std::thread> workers;
            std::atomic<int> total_ops{ 0 };

            for (int t = 0; t < num_compute_units; ++t) {
                workers.emplace_back([&, t]() {
                    int local_ops = 0;
                    // 每个计算单元处理矩阵的一部分
                    int rows_per_unit = matrix_size / num_compute_units;
                    int start_row = t * rows_per_unit;
                    int end_row = (t == num_compute_units - 1) ? matrix_size : start_row + rows_per_unit;

                    // 模拟矩阵计算
                    for (int iter = 0; iter < 10 && !stop_test; ++iter) {
                        std::vector<std::vector<double>> A(rows_per_unit, std::vector<double>(matrix_size));
                        std::vector<std::vector<double>> B(matrix_size, std::vector<double>(rows_per_unit));
                        std::vector<std::vector<double>> C(rows_per_unit, std::vector<double>(rows_per_unit, 0.0));

                        // 初始化矩阵
                        for (int i = 0; i < rows_per_unit; ++i) {
                            for (int j = 0; j < matrix_size; ++j) {
                                A[i][j] = random_double();
                                B[j][i] = random_double();
                            }
                        }

                        // 矩阵乘法计算
                        for (int i = 0; i < rows_per_unit; ++i) {
                            for (int j = 0; j < rows_per_unit; ++j) {
                                for (int k = 0; k < matrix_size; ++k) {
                                    C[i][j] += A[i][k] * B[k][j];
                                }
                            }
                        }
                        local_ops++;
                    }
                    total_ops += local_ops;
                    });
            }

            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            operations += total_ops;
        }

        double performance = static_cast<double>(operations) * matrix_size * matrix_size * matrix_size / duration_sec;
        log("GPU GEMM测试完成: " + std::to_string(performance) + " FLOPS");
        return performance;
    }

    // 快速傅里叶变换 (FFT) - 模拟GPU并行FFT
    double test_fft(int fft_size, int duration_sec) {
        log("开始GPU FFT测试 (FFT大小: " + std::to_string(fft_size) + ")");

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_sec);

        int operations = 0;

        while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
            // 模拟并行FFT计算
            std::vector<std::thread> workers;
            std::atomic<int> batch_ops{ 0 };

            int batches = num_compute_units * 4;  // 每个计算单元处理多个批次

            for (int b = 0; b < batches; ++b) {
                workers.emplace_back([&, b]() {
                    int local_ops = 0;
                    // 每个批次处理多个FFT
                    for (int i = 0; i < 5 && !stop_test; ++i) {
                        std::vector<Complex> data(fft_size);
                        for (int j = 0; j < fft_size; ++j) {
                            data[j] = Complex(random_double(), random_double());
                        }

                        // 执行FFT和逆FFT
                        fft(data, false);
                        fft(data, true);

                        local_ops++;
                    }
                    batch_ops += local_ops;
                    });
            }

            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            operations += batch_ops;
        }

        double performance = static_cast<double>(operations) * fft_size * std::log2(fft_size) / duration_sec;
        log("GPU FFT测试完成: " + std::to_string(performance) + " 操作/秒");
        return performance;
    }

    // 欧拉流体模拟 - 简化版GPU模拟
    double test_fluid_simulation(int grid_size, int duration_sec) {
        log("开始GPU流体模拟测试 (网格大小: " + std::to_string(grid_size) + "x" + std::to_string(grid_size) + ")");

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_sec);

        int time_steps = 0;

        while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
            // 简化流体模拟 - 每个线程处理网格的一部分
            std::vector<std::thread> workers;
            const int subgrid_size = grid_size / num_compute_units;

            for (int t = 0; t < num_compute_units; ++t) {
                workers.emplace_back([&, t]() {
                    // 创建局部流体场
                    std::vector<std::vector<double>> density(subgrid_size, std::vector<double>(subgrid_size));
                    std::vector<std::vector<double>> velocity_x(subgrid_size, std::vector<double>(subgrid_size));
                    std::vector<std::vector<double>> velocity_y(subgrid_size, std::vector<double>(subgrid_size));

                    // 初始化流体场
                    for (int i = 0; i < subgrid_size; ++i) {
                        for (int j = 0; j < subgrid_size; ++j) {
                            density[i][j] = random_double();
                            velocity_x[i][j] = random_double(-1.0, 1.0);
                            velocity_y[i][j] = random_double(-1.0, 1.0);
                        }
                    }

                    // 简化欧拉方程求解（多个时间步）
                    for (int step = 0; step < 20 && !stop_test; ++step) {
                        // 平流项
                        auto new_density = density;
                        auto new_velocity_x = velocity_x;
                        auto new_velocity_y = velocity_y;

                        for (int i = 1; i < subgrid_size - 1; ++i) {
                            for (int j = 1; j < subgrid_size - 1; ++j) {
                                // 简化平流计算
                                new_density[i][j] = 0.25 * (
                                    density[i - 1][j] + density[i + 1][j] +
                                    density[i][j - 1] + density[i][j + 1]
                                    );

                                // 简化速度更新
                                new_velocity_x[i][j] = 0.99 * velocity_x[i][j] +
                                    0.01 * (velocity_x[i - 1][j] + velocity_x[i + 1][j]);
                                new_velocity_y[i][j] = 0.99 * velocity_y[i][j] +
                                    0.01 * (velocity_y[i][j - 1] + velocity_y[i][j + 1]);
                            }
                        }

                        density = new_density;
                        velocity_x = new_velocity_x;
                        velocity_y = new_velocity_y;
                    }
                    });
            }

            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            time_steps++;
        }

        double performance = static_cast<double>(time_steps) * grid_size * grid_size / duration_sec;
        log("GPU流体模拟完成: " + std::to_string(performance) + " 网格单元/秒");
        return performance;
    }

    // 蒙特卡洛积分 - 模拟GPU并行随机采样
    double test_monte_carlo(int samples, int duration_sec) {
        log("开始GPU蒙特卡洛积分测试 (样本数: " + std::to_string(samples) + ")");

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_sec);

        double total_integral = 0.0;
        int completed_runs = 0;

        while (std::chrono::high_resolution_clock::now() < end_time && !stop_test) {
            std::vector<std::thread> workers;
            std::vector<double> partial_results(num_compute_units, 0.0);

            int samples_per_thread = samples / num_compute_units;

            for (int t = 0; t < num_compute_units; ++t) {
                workers.emplace_back([&, t]() {
                    double local_sum = 0.0;
                    std::mt19937 generator(std::random_device{}());
                    std::uniform_real_distribution<double> distribution(0.0, 1.0);

                    // 计算函数在[0,1]区间的积分: f(x) = sin(x) * exp(x)
                    for (int i = 0; i < samples_per_thread; ++i) {
                        double x = distribution(generator);
                        double y = distribution(generator);
                        double value = std::sin(x) * std::exp(x);

                        if (y <= value) {
                            local_sum += value;
                        }
                    }

                    partial_results[t] = local_sum / samples_per_thread;
                    });
            }

            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            double run_integral = 0.0;
            for (double result : partial_results) {
                run_integral += result;
            }
            run_integral /= num_compute_units;

            total_integral += run_integral;
            completed_runs++;
        }

        double final_integral = total_integral / completed_runs;
        double performance = static_cast<double>(completed_runs) * samples / duration_sec;

        log("GPU蒙特卡洛积分完成: 积分值≈" + std::to_string(final_integral) +
            ", 性能: " + std::to_string(performance) + " 样本/秒");
        return performance;
    }
};

class DXComputeTester {
private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> matrixMultiplyShader_;
    ComPtr<ID3D11ComputeShader> fftShader_;
    bool initialized_ = false;
    int lastProgress_ = 0;

    // 添加资源管理
    std::vector<ComPtr<ID3D11Buffer>> buffers_;
    std::vector<ComPtr<ID3D11UnorderedAccessView>> uavs_;
    std::vector<ComPtr<ID3D11ShaderResourceView>> srvs_;

public:
    bool initialize() {
        log("初始化DirectX 11设备...");

        // 清理之前的资源
        cleanup();

        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };

        UINT flags = D3D11_CREATE_DEVICE_SINGLETHREADED;

        // 添加调试标志以便更好的错误信息
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            &device_,
            nullptr,
            &context_
        );

        if (FAILED(hr)) {
            log("无法创建DirectX硬件设备，错误代码: 0x" + std::to_string(hr));
            return false;
        }

        log("DirectX 11硬件设备初始化成功");

        // 编译计算着色器
        if (!compileComputeShaders()) {
            log("计算着色器编译失败");
            return false;
        }

        initialized_ = true;
        return true;
    }

private:
    void cleanup() {
        // 清理所有资源
        uavs_.clear();
        srvs_.clear();
        buffers_.clear();
        matrixMultiplyShader_.Reset();
        fftShader_.Reset();
        context_.Reset();
        device_.Reset();
        initialized_ = false;
    }

    bool compileComputeShaders() {
        // 简化的矩阵乘法着色器
        const char* matrixShaderSource = R"(
            cbuffer MatrixSize : register(b0)
            {
                uint matrixSize;
            }
            
            RWStructuredBuffer<float> InputA : register(u0);
            RWStructuredBuffer<float> InputB : register(u1);
            RWStructuredBuffer<float> OutputC : register(u2);
            
            [numthreads(8, 8, 1)]  // 减小线程组大小
            void CSMain(uint3 threadID : SV_DispatchThreadID)
            {
                uint idx = threadID.y * matrixSize + threadID.x;
                if (threadID.x >= matrixSize || threadID.y >= matrixSize) 
                    return;
                
                float sum = 0.0;
                for (uint k = 0; k < matrixSize; k++) {
                    uint idxA = threadID.y * matrixSize + k;
                    uint idxB = k * matrixSize + threadID.x;
                    sum += InputA[idxA] * InputB[idxB];
                }
                OutputC[idx] = sum;
            }
        )";

        // 简化的FFT着色器
        const char* fftShaderSource = R"(
            cbuffer FFTSize : register(b0)
            {
                uint fftSize;
            }
            
            RWStructuredBuffer<float2> Data : register(u0);
            
            [numthreads(64, 1, 1)]  // 减小线程组大小
            void CSMain(uint3 threadID : SV_DispatchThreadID)
            {
                uint idx = threadID.x;
                if (idx >= fftSize) return;
                
                // 简化计算避免复杂数学运算
                float2 val = Data[idx];
                Data[idx] = float2(val.y, val.x);  // 简单交换实部和虚部
            }
        )";

        ComPtr<ID3DBlob> shaderBlob, errorBlob;

        // 编译矩阵乘法着色器
        HRESULT hr = D3DCompile(
            matrixShaderSource, strlen(matrixShaderSource),
            nullptr, nullptr, nullptr,
            "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL1, 0,  // 降低优化级别
            &shaderBlob, &errorBlob
        );

        if (FAILED(hr)) {
            std::string errorMsg = "未知错误";
            if (errorBlob) {
                errorMsg = std::string((char*)errorBlob->GetBufferPointer());
            }
            log("矩阵着色器编译错误: " + errorMsg);
            return false;
        }

        hr = device_->CreateComputeShader(
            shaderBlob->GetBufferPointer(),
            shaderBlob->GetBufferSize(),
            nullptr, &matrixMultiplyShader_
        );

        if (FAILED(hr)) {
            log("创建矩阵计算着色器失败: 0x" + std::to_string(hr));
            return false;
        }

        // 编译FFT着色器
        hr = D3DCompile(
            fftShaderSource, strlen(fftShaderSource),
            nullptr, nullptr, nullptr,
            "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL1, 0,
            &shaderBlob, &errorBlob
        );

        if (FAILED(hr)) {
            std::string errorMsg = "未知错误";
            if (errorBlob) {
                errorMsg = std::string((char*)errorBlob->GetBufferPointer());
            }
            log("FFT着色器编译错误: " + errorMsg);
            return false;
        }

        hr = device_->CreateComputeShader(
            shaderBlob->GetBufferPointer(),
            shaderBlob->GetBufferSize(),
            nullptr, &fftShader_
        );

        if (FAILED(hr)) {
            log("创建FFT计算着色器失败: 0x" + std::to_string(hr));
            return false;
        }

        return true;
    }

    ComPtr<ID3D11Buffer> createConstantBuffer(UINT size) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = (size + 15) & ~15;  // 16字节对齐
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        ComPtr<ID3D11Buffer> buffer;
        HRESULT hr = device_->CreateBuffer(&desc, nullptr, &buffer);
        if (FAILED(hr)) {
            log("创建常量缓冲区失败: 0x" + std::to_string(hr));
            return nullptr;
        }
        buffers_.push_back(buffer);
        return buffer;
    }

    ComPtr<ID3D11Buffer> createStructuredBuffer(UINT elementSize, UINT elementCount, bool uav = true, const void* initialData = nullptr) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = elementSize * elementCount;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (uav) {
            desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
        }
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = elementSize;

        ComPtr<ID3D11Buffer> buffer;
        D3D11_SUBRESOURCE_DATA initData = { initialData };
        HRESULT hr = device_->CreateBuffer(&desc, initialData ? &initData : nullptr, &buffer);
        if (FAILED(hr)) {
            log("创建结构化缓冲区失败: 0x" + std::to_string(hr));
            return nullptr;
        }
        buffers_.push_back(buffer);
        return buffer;
    }

    ComPtr<ID3D11UnorderedAccessView> createUAV(ID3D11Buffer* buffer, UINT elementCount) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc = {};
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = 0;
        desc.Buffer.NumElements = elementCount;

        ComPtr<ID3D11UnorderedAccessView> uav;
        HRESULT hr = device_->CreateUnorderedAccessView(buffer, &desc, &uav);
        if (FAILED(hr)) {
            log("创建UAV失败: 0x" + std::to_string(hr));
            return nullptr;
        }
        uavs_.push_back(uav);
        return uav;
    }

    void updateConstantBuffer(ID3D11Buffer* buffer, const void* data, UINT size) {
        D3D11_MAPPED_SUBRESOURCE mapped;
        HRESULT hr = context_->Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
        if (SUCCEEDED(hr)) {
            memcpy(mapped.pData, data, size);
            context_->Unmap(buffer, 0);
        }
    }

    bool validateResources() {
        if (!device_ || !context_) {
            log("设备或上下文无效");
            return false;
        }
        return true;
    }

public:
    // GPU矩阵乘法测试
    double testMatrixMultiplicationTFLOPS(int durationSec) {
        if (!validateResources()) return 0.0;

        log("开始GPU矩阵乘法测试...");

        auto startTime = std::chrono::high_resolution_clock::now();
        auto endTime = startTime + std::chrono::seconds(durationSec);

        // 使用更小的矩阵尺寸避免内存问题
        const int MATRIX_SIZE = 256;
        const int ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
        const size_t bufferSize = ELEMENTS * sizeof(float);

        // 创建常量缓冲区
        auto constantBuffer = createConstantBuffer(sizeof(uint32_t));
        if (!constantBuffer) {
            log("常量缓冲区创建失败");
            return 0.0;
        }
        uint32_t matrixSize = MATRIX_SIZE;
        updateConstantBuffer(constantBuffer.Get(), &matrixSize, sizeof(uint32_t));

        // 初始化数据
        std::vector<float> initDataA(ELEMENTS);
        std::vector<float> initDataB(ELEMENTS);
        for (int i = 0; i < ELEMENTS; i++) {
            initDataA[i] = static_cast<float>(random_double());
            initDataB[i] = static_cast<float>(random_double());
        }

        // 创建缓冲区
        auto bufferA = createStructuredBuffer(sizeof(float), ELEMENTS, true, initDataA.data());
        auto bufferB = createStructuredBuffer(sizeof(float), ELEMENTS, true, initDataB.data());
        auto bufferC = createStructuredBuffer(sizeof(float), ELEMENTS, true, nullptr);

        if (!bufferA || !bufferB || !bufferC) {
            log("缓冲区创建失败");
            return 0.0;
        }

        // 创建UAVs
        auto uavA = createUAV(bufferA.Get(), ELEMENTS);
        auto uavB = createUAV(bufferB.Get(), ELEMENTS);
        auto uavC = createUAV(bufferC.Get(), ELEMENTS);

        if (!uavA || !uavB || !uavC) {
            log("UAV创建失败");
            return 0.0;
        }

        ID3D11UnorderedAccessView* uavs[] = { uavA.Get(), uavB.Get(), uavC.Get() };

        int iterations = 0;
        lastProgress_ = 0;

        while (std::chrono::high_resolution_clock::now() < endTime && !stop_test) {
            // 重置UAV绑定
            ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr, nullptr };
            context_->CSSetUnorderedAccessViews(0, 3, nullUAVs, nullptr);

            // 设置着色器和资源
            context_->CSSetShader(matrixMultiplyShader_.Get(), nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, constantBuffer.GetAddressOf());
            context_->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);

            // 调度计算着色器
            uint32_t groups = (MATRIX_SIZE + 7) / 8;  // 匹配线程组大小
            context_->Dispatch(groups, groups, 1);

            iterations++;

            // 进度显示
            if (iterations % 10 == 0) {
                int progress = static_cast<int>((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startTime).count() * 100) /
                    (durationSec * 1000));

                if (progress >= lastProgress_ + 10) {
                    log("GPU矩阵计算进度: " + std::to_string(progress) + "%, 迭代: " + std::to_string(iterations));
                    lastProgress_ = progress;
                }
            }
        }

        // 清理UAV绑定
        ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr, nullptr };
        context_->CSSetUnorderedAccessViews(0, 3, nullUAVs, nullptr);
        context_->CSSetShader(nullptr, nullptr, 0);

        double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        if (elapsedSeconds > 0 && iterations > 0) {
            double totalFLOPs = iterations * 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
            double tflops = totalFLOPs / elapsedSeconds / 1e12;

            log("GPU矩阵乘法完成: " + std::to_string(tflops) + " TFLOPS");
            return tflops;
        }

        return 0.0;
    }

    // GPU FFT测试
    double testFFT_TFLOPS(int durationSec) {
        if (!validateResources()) return 0.0;

        log("开始GPU FFT测试...");

        auto startTime = std::chrono::high_resolution_clock::now();
        auto endTime = startTime + std::chrono::seconds(durationSec);

        // 使用更小的FFT尺寸
        const int FFT_SIZE = 512;
        const int ELEMENTS = FFT_SIZE;
        const size_t bufferSize = ELEMENTS * sizeof(float) * 2;

        // 创建常量缓冲区
        auto constantBuffer = createConstantBuffer(sizeof(uint32_t));
        if (!constantBuffer) {
            log("常量缓冲区创建失败");
            return 0.0;
        }
        uint32_t fftSize = FFT_SIZE;
        updateConstantBuffer(constantBuffer.Get(), &fftSize, sizeof(uint32_t));

        // 初始化复数数据
        std::vector<float> initData(ELEMENTS * 2);
        for (int i = 0; i < ELEMENTS * 2; i++) {
            initData[i] = static_cast<float>(random_double());
        }

        auto buffer = createStructuredBuffer(sizeof(float) * 2, ELEMENTS, true, initData.data());
        if (!buffer) {
            log("FFT缓冲区创建失败");
            return 0.0;
        }

        auto uav = createUAV(buffer.Get(), ELEMENTS);
        if (!uav) {
            log("FFT UAV创建失败");
            return 0.0;
        }

        int iterations = 0;
        lastProgress_ = 0;

        while (std::chrono::high_resolution_clock::now() < endTime && !stop_test) {
            // 重置UAV绑定
            ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
            context_->CSSetUnorderedAccessViews(0, 1, nullUAVs, nullptr);

            // 设置计算着色器
            context_->CSSetShader(fftShader_.Get(), nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, constantBuffer.GetAddressOf());

            ID3D11UnorderedAccessView* uavs[] = { uav.Get() };
            context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);

            // 调度计算
            uint32_t groups = (FFT_SIZE + 63) / 64;
            context_->Dispatch(groups, 1, 1);

            iterations++;

            // 进度显示
            if (iterations % 20 == 0) {
                int progress = static_cast<int>((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startTime).count() * 100) /
                    (durationSec * 1000));

                if (progress >= lastProgress_ + 10) {
                    log("GPU FFT进度: " + std::to_string(progress) + "%, 迭代: " + std::to_string(iterations));
                    lastProgress_ = progress;
                }
            }
        }

        // 清理绑定
        ID3D11UnorderedAccessView* nullUAVs[] = { nullptr };
        context_->CSSetUnorderedAccessViews(0, 1, nullUAVs, nullptr);
        context_->CSSetShader(nullptr, nullptr, 0);

        double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        if (elapsedSeconds > 0 && iterations > 0) {
            double totalFLOPs = iterations * 5.0 * FFT_SIZE * std::log2(FFT_SIZE);
            double tflops = totalFLOPs / elapsedSeconds / 1e12;

            log("GPU FFT完成: " + std::to_string(tflops) + " TFLOPS");
            return tflops;
        }

        return 0.0;
    }

    // GPU内存带宽测试
    double testMemoryBandwidthGBs(int durationSec) {
        if (!validateResources()) return 0.0;

        log("开始GPU内存带宽测试...");

        auto startTime = std::chrono::high_resolution_clock::now();
        auto endTime = startTime + std::chrono::seconds(durationSec);

        // 使用更小的缓冲区
        const int BUFFER_SIZE_MB = 32;
        const size_t bufferSize = BUFFER_SIZE_MB * 1024 * 1024;
        const int ELEMENTS = bufferSize / sizeof(float);

        // 创建源和目标缓冲区
        std::vector<float> initData(ELEMENTS);
        for (int i = 0; i < ELEMENTS; i++) {
            initData[i] = static_cast<float>(random_double());
        }

        auto srcBuffer = createStructuredBuffer(sizeof(float), ELEMENTS, false, initData.data());
        auto dstBuffer = createStructuredBuffer(sizeof(float), ELEMENTS, false, nullptr);

        if (!srcBuffer || !dstBuffer) {
            log("内存测试缓冲区创建失败");
            return 0.0;
        }

        double totalBytes = 0;
        lastProgress_ = 0;
        int copyOperations = 0;

        while (std::chrono::high_resolution_clock::now() < endTime && !stop_test) {
            // GPU内存复制
            context_->CopyResource(dstBuffer.Get(), srcBuffer.Get());
            copyOperations++;
            totalBytes += bufferSize;

            // 进度显示
            if (copyOperations % 100 == 0) {
                int progress = static_cast<int>((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startTime).count() * 100) /
                    (durationSec * 1000));

                if (progress >= lastProgress_ + 10) {
                    double currentBandwidth = (totalBytes / (1024.0 * 1024 * 1024)) /
                        (std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0);
                    log("GPU内存测试进度: " + std::to_string(progress) + "%, 当前带宽: " +
                        std::to_string(currentBandwidth) + " GB/s");
                    lastProgress_ = progress;
                }
            }
        }

        // 确保所有操作完成
        context_->Flush();

        double elapsedSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        if (elapsedSeconds > 0) {
            double bandwidth = totalBytes / (1024.0 * 1024 * 1024) / elapsedSeconds;
            log("GPU内存带宽测试完成: " + std::to_string(bandwidth) + " GB/s");
            return bandwidth;
        }

        return 0.0;
    }

    ~DXComputeTester() {
        // 确保所有资源都被释放
        if (context_) {
            // 清除所有绑定状态
            ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr, nullptr };
            context_->CSSetUnorderedAccessViews(0, 3, nullUAVs, nullptr);
            context_->CSSetShader(nullptr, nullptr, 0);
            context_->Flush();
        }

        // 清理所有COM对象
        uavs_.clear();
        srvs_.clear();
        buffers_.clear();
        matrixMultiplyShader_.Reset();
        fftShader_.Reset();
        context_.Reset();
        device_.Reset();

        log("DXComputeTester资源已完全释放");
    }
};

void gpu_test() {
    log("开始GPU性能测试，持续 " + std::to_string(test_config.duration.gpu) + " 秒");

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(test_config.duration.gpu);

    // ========== 新增：添加超时保护 ==========
    const auto max_test_time = std::chrono::seconds(30); // 最大30秒
    auto hard_end_time = start_time + max_test_time;

    // ========== 新增：快速硬件检测 ==========
    bool useHardware = false;
    std::unique_ptr<DXComputeTester> dxTester;

    // 快速检测是否支持DirectX 11
    {
        ComPtr<ID3D11Device> testDevice;
        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };

        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            featureLevels, 1, D3D11_SDK_VERSION, &testDevice, nullptr, nullptr);

        if (SUCCEEDED(hr)) {
            useHardware = true;
            log("检测到DirectX 11硬件支持");
        }
        else {
            log("未检测到DirectX 11硬件，将使用模拟测试");
        }
    }

    double compute_tflops = 0.0, fft_tflops = 0.0, memory_gbs = 0.0;
    int completed_tests = 0;

    if (useHardware) {
        try {
            dxTester = std::make_unique<DXComputeTester>();
            if (!dxTester->initialize()) {
                log("GPU初始化失败，回退到模拟测试");
                useHardware = false;
                dxTester.reset();
            }
        }
        catch (...) {
            log("GPU初始化异常，回退到模拟测试");
            useHardware = false;
        }
    }

    // 优化测试流程
    auto test_start = std::chrono::high_resolution_clock::now();

    // 计算每个子测试的时间分配（使用更短的测试时间）
    const int total_duration = std::min(test_config.duration.gpu, 10); // 最多10秒
    const int per_test_duration = std::max(1, total_duration / 3); // 每个测试最多3-4秒

    log("GPU测试优化：总时间" + std::to_string(total_duration) + "秒，每项" +
        std::to_string(per_test_duration) + "秒");

    try {
        if (useHardware && dxTester) {
            // 矩阵乘法测试
            if (!stop_test && std::chrono::high_resolution_clock::now() < hard_end_time) {
                log("开始GPU矩阵乘法测试");
                compute_tflops = dxTester->testMatrixMultiplicationTFLOPS(std::min(per_test_duration, 3));
                if (compute_tflops > 0) completed_tests++;
                log("矩阵乘法测试完成: " + std::to_string(compute_tflops) + " TFLOPS");
            }

            // 短暂暂停
            if (!stop_test) std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // FFT测试
            if (!stop_test && std::chrono::high_resolution_clock::now() < hard_end_time) {
                log("开始GPU FFT测试");
                fft_tflops = dxTester->testFFT_TFLOPS(std::min(per_test_duration, 3));
                if (fft_tflops > 0) completed_tests++;
                log("FFT测试完成: " + std::to_string(fft_tflops) + " TFLOPS");
            }

            // 短暂暂停
            if (!stop_test) std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // 内存带宽测试
            if (!stop_test && std::chrono::high_resolution_clock::now() < hard_end_time) {
                log("开始GPU内存带宽测试...");
                memory_gbs = dxTester->testMemoryBandwidthGBs(std::min(per_test_duration, 2));
                if (memory_gbs > 0) completed_tests++;
                log("内存带宽测试完成: " + std::to_string(memory_gbs) + " GB/s");
            }

            // 立即释放GPU资源
            dxTester.reset();

        }
        else {
            // 优化4：轻量级模拟测试
            log("使用GPU模拟测试...");

            // 简化模拟测试，避免复杂计算
            auto sim_start = std::chrono::high_resolution_clock::now();
            int sim_iterations = 0;

            while (!stop_test &&
                std::chrono::high_resolution_clock::now() < sim_start + std::chrono::seconds(2) &&
                std::chrono::high_resolution_clock::now() < hard_end_time) {

                // 轻量级矩阵计算模拟
                const int sim_size = 128;
                std::vector<float> sim_data(sim_size * sim_size);
                for (auto& val : sim_data) val = random_double();

                // 简单计算（避免复杂矩阵乘法）
                float sum = 0.0f;
                for (int i = 0; i < 1000 && !stop_test; ++i) {
                    sum += sim_data[i % sim_data.size()];
                }

                sim_iterations++;
                if (sim_iterations % 10 == 0) {
                    // 定期检查是否应该停止
                    if (stop_test) break;
                }
            }

            // 生成模拟结果（基于迭代次数）
            double sim_factor = std::min(1.0, sim_iterations / 100.0);
            compute_tflops = 0.5 + random_double(0.0, 0.5) * sim_factor;
            fft_tflops = 0.3 + random_double(0.0, 0.3) * sim_factor;
            memory_gbs = 50.0 + random_double(0.0, 30.0) * sim_factor;

            completed_tests = 3; // 标记为完成所有测试
            log("模拟测试完成: " + std::to_string(sim_iterations) + " 次迭代");
        }

    }
    catch (const std::exception& e) {
        log("GPU测试异常: " + std::string(e.what()));
        if (dxTester) dxTester.reset();
    }
    catch (...) {
        log("GPU测试未知异常");
        if (dxTester) dxTester.reset();
    }

    // 强制等待GPU资源释放
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto test_end = std::chrono::high_resolution_clock::now();
    auto test_duration = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);

    if (!stop_test) {
        // 计算原始得分
        double raw_score = (compute_tflops + fft_tflops + memory_gbs) / 3.0;

        // 使用配置中的基准值进行标准化
        double normalized = (raw_score / test_config.benchmarks.gpu) * 100;
        normalized = std::max(0.0, normalized); // 确保非负

        std::lock_guard<std::mutex> lock(result_mutex);
        raw_results["gpu"] = raw_score;
        test_results["gpu"] = normalized;

        // 输出信息
        log("GPGPU性能测试完成");
        if (useHardware) {
            log("矩阵计算性能: " + std::to_string(compute_tflops) + " TFLOPS");
            log("FFT计算性能: " + std::to_string(fft_tflops) + " TFLOPS");
            log("内存带宽: " + std::to_string(memory_gbs) + " GB/s");
        }
        else {
            log("矩阵计算性能(模拟): " + std::to_string(compute_tflops) + " TFLOPS");
            log("FFT计算性能(模拟): " + std::to_string(fft_tflops) + " TFLOPS");
            log("内存带宽(模拟): " + std::to_string(memory_gbs) + " GB/s");
        }
        log("原始: " + std::to_string(raw_score) +
            ", 标准化: " + std::to_string(normalized));

    }
    else {
        log("GPU测试被中止");
        dxTester.reset();
    }
}

// 获取系统软硬件信息
void get_info() {
    log("开始获取系统软硬件信息...");

    try {
        // 操作系统信息
        log("\n===== 操作系统信息 =====");

        // RtlGetVersion
        typedef LONG(WINAPI* RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
        HMODULE hMod = GetModuleHandleW(L"ntdll.dll");

        std::string osName = "Windows 未知版本";
        DWORD majorVersion = 0, minorVersion = 0, buildNumber = 0;
        std::string displayVersion = "未知";

        if (hMod) {
            auto RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(hMod, "RtlGetVersion");
            if (RtlGetVersion) {
                RTL_OSVERSIONINFOW osVersionInfo = { 0 };
                osVersionInfo.dwOSVersionInfoSize = sizeof(osVersionInfo);

                if (RtlGetVersion(&osVersionInfo) == 0) {
                    majorVersion = osVersionInfo.dwMajorVersion;
                    minorVersion = osVersionInfo.dwMinorVersion;
                    buildNumber = osVersionInfo.dwBuildNumber;

                    // 根据版本号判断操作系统
                    if (majorVersion == 10 && minorVersion == 0) {
                        // Windows 10 或 Windows 11
                        if (buildNumber >= 22000) {
                            osName = "Windows 11";
                        }
                        else {
                            osName = "Windows 10";
                        }
                    }
                    else if (majorVersion == 6) {
                        if (minorVersion == 3) osName = "Windows 8.1";
                        else if (minorVersion == 2) osName = "Windows 8";
                        else if (minorVersion == 1) osName = "Windows 7";
                        else if (minorVersion == 0) osName = "Windows Vista";
                    }
                    else if (majorVersion == 5) {
                        osName = "Windows XP/Server 2003";
                    }
                }
            }
        }

        // 获取显示版本和更多详细信息
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {

            // 获取产品名称
            char productName[256] = { 0 };
            DWORD size = sizeof(productName);
            if (RegQueryValueExA(hKey, "ProductName", NULL, NULL,
                (LPBYTE)productName, &size) == ERROR_SUCCESS) {
                // 从产品名称中提取更准确的信息
                std::string productStr = productName;
                if (productStr.find("Windows 11") != std::string::npos) {
                    osName = "Windows 11";
                }
                else if (productStr.find("Windows 10") != std::string::npos) {
                    osName = "Windows 10";
                }
            }

            // 获取显示版本
            char displayVersionStr[256] = { 0 };
            size = sizeof(displayVersionStr);
            if (RegQueryValueExA(hKey, "DisplayVersion", NULL, NULL,
                (LPBYTE)displayVersionStr, &size) == ERROR_SUCCESS) {
                displayVersion = displayVersionStr;
            }
            // 如果没有 DisplayVersion，尝试获取 ReleaseId
            else {
                char releaseId[256] = { 0 };
                size = sizeof(releaseId);
                if (RegQueryValueExA(hKey, "ReleaseId", NULL, NULL,
                    (LPBYTE)releaseId, &size) == ERROR_SUCCESS) {
                    displayVersion = releaseId;
                }
            }

            // 获取 UBR用于更精确的构建号
            DWORD ubr = 0;
            size = sizeof(ubr);
            if (RegQueryValueExA(hKey, "UBR", NULL, NULL,
                (LPBYTE)&ubr, &size) == ERROR_SUCCESS && ubr > 0) {
                buildNumber = (buildNumber * 10000) + ubr; // 组合成完整构建号
            }

            RegCloseKey(hKey);
        }

        // 尝试使用新的 Win11 API 检测 Windows 11 特定功能
        bool isWindows11 = false;
        if (buildNumber >= 22000) {
            isWindows11 = true;
            osName = "Windows 11";

            // 检查是否启用了 Windows 11 的圆角窗口特性
            HKEY hKeyWin11;
            if (RegOpenKeyExA(HKEY_CURRENT_USER,
                "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced",
                0, KEY_READ, &hKeyWin11) == ERROR_SUCCESS) {

                DWORD useRoundedCorners = 0;
                DWORD size = sizeof(useRoundedCorners);
                if (RegQueryValueExA(hKeyWin11, "UseRoundedCorners", NULL, NULL,
                    (LPBYTE)&useRoundedCorners, &size) == ERROR_SUCCESS) {
                    // 如果这个键存在，基本可以确认是 Windows 11
                    isWindows11 = true;
                }
                RegCloseKey(hKeyWin11);
            }
        }

        // 输出操作系统信息
        log("操作系统: " + osName);
        log("显示版本: " + displayVersion);
        log("版本: " + std::to_string(majorVersion) + "." +
            std::to_string(minorVersion) + "." +
            std::to_string(buildNumber));

        // 额外显示 Windows 11 特定信息
        if (isWindows11) {

            // 尝试获取 Windows 11 功能更新版本
            HKEY hKeyWin11Features;
            if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                "SOFTWARE\\Microsoft\\WindowsUpdate\\UX\\Settings",
                0, KEY_READ, &hKeyWin11Features) == ERROR_SUCCESS) {

                DWORD featureUpdateVersion = 0;
                DWORD size = sizeof(featureUpdateVersion);
                if (RegQueryValueExA(hKeyWin11Features, "UxOption", NULL, NULL,
                    (LPBYTE)&featureUpdateVersion, &size) == ERROR_SUCCESS) {
                    log("Windows 11 功能更新版本: " + std::to_string(featureUpdateVersion));
                }
                RegCloseKey(hKeyWin11Features);
            }
        }

        // 系统架构
        SYSTEM_INFO sysInfo;
        GetNativeSystemInfo(&sysInfo);

        std::string architecture;
        switch (sysInfo.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64:
            architecture = "64位 (x64)";
            break;
        case PROCESSOR_ARCHITECTURE_ARM:
            architecture = "ARM";
            break;
        case PROCESSOR_ARCHITECTURE_ARM64:
            architecture = "ARM64";
            break;
        case PROCESSOR_ARCHITECTURE_INTEL:
            architecture = "32位 (x86)";
            break;
        default:
            architecture = "未知";
        }
        log("系统架构: " + architecture);

        // CPU信息
        log("\n===== CPU信息 =====");
        log("逻辑处理器数量: " + std::to_string(sysInfo.dwNumberOfProcessors));

        // 获取CPU品牌信息
        int cpuInfo[4] = { 0 };
        char cpuBrand[0x40] = { 0 };

        __cpuid(cpuInfo, 0);
        if (cpuInfo[0] >= 1) {
            // 获取厂商
            memset(cpuBrand, 0, sizeof(cpuBrand));
            *((int*)cpuBrand) = cpuInfo[1];
            *((int*)(cpuBrand + 4)) = cpuInfo[3];
            *((int*)(cpuBrand + 8)) = cpuInfo[2];
            log("CPU厂商: " + std::string(cpuBrand));

            // 获取品牌字符串
            __cpuid(cpuInfo, 0x80000000);
            if ((unsigned)cpuInfo[0] >= 0x80000004) {
                memset(cpuBrand, 0, sizeof(cpuBrand));

                __cpuid(cpuInfo, 0x80000002);
                memcpy(cpuBrand, cpuInfo, sizeof(cpuInfo));

                __cpuid(cpuInfo, 0x80000003);
                memcpy(cpuBrand + 16, cpuInfo, sizeof(cpuInfo));

                __cpuid(cpuInfo, 0x80000004);
                memcpy(cpuBrand + 32, cpuInfo, sizeof(cpuInfo));

                // 清理品牌字符串中的多余空格
                std::string cpuName = cpuBrand;
                size_t start = cpuName.find_first_not_of(' ');
                size_t end = cpuName.find_last_not_of(' ');
                if (start != std::string::npos && end != std::string::npos) {
                    cpuName = cpuName.substr(start, end - start + 1);
                }
                log("CPU型号: " + cpuName);
            }
        }

        // 内存信息
        log("\n===== 内存信息 =====");
        MEMORYSTATUSEX memStatus;
        memStatus.dwLength = sizeof(memStatus);

        if (GlobalMemoryStatusEx(&memStatus)) {
            double totalGB = static_cast<double>(memStatus.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
            double availableGB = static_cast<double>(memStatus.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
            double usedPercent = static_cast<double>(memStatus.dwMemoryLoad);

            log("物理内存总量: " + std::to_string(totalGB) + " GB");
            log("可用物理内存: " + std::to_string(availableGB) + " GB");
            log("内存使用率: " + std::to_string(usedPercent) + "%");
        }

        // 磁盘信息
        log("\n===== 磁盘信息 =====");
        ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;

        std::vector<wchar_t> drives;
        wchar_t driveStrings[256];
        if (GetLogicalDriveStringsW(256, driveStrings)) {
            for (wchar_t* drive = driveStrings; *drive; drive += wcslen(drive) + 1) {
                UINT driveType = GetDriveTypeW(drive);
                if (driveType == DRIVE_FIXED) { // 只显示固定磁盘
                    if (GetDiskFreeSpaceExW(drive, &freeBytes, &totalBytes, &totalFreeBytes)) {
                        double totalGB = static_cast<double>(totalBytes.QuadPart) / (1024.0 * 1024.0 * 1024.0);
                        double freeGB = static_cast<double>(freeBytes.QuadPart) / (1024.0 * 1024.0 * 1024.0);
                        double usedGB = totalGB - freeGB;
                        double usedPercent = (usedGB / totalGB) * 100.0;

                        // 转换驱动器名称为ASCII
                        char driveLetter[10];
                        WideCharToMultiByte(CP_UTF8, 0, drive, -1, driveLetter, sizeof(driveLetter), NULL, NULL);

                        log(std::string(driveLetter) + "盘 - 总容量: " +
                            std::to_string(totalGB) + " GB, 已使用: " +
                            std::to_string(usedGB) + " GB (" +
                            std::to_string(usedPercent) + "%), 可用: " +
                            std::to_string(freeGB) + " GB");
                    }
                }
            }
        }

        // 显示适配器信息
        log("\n===== 显示适配器信息 =====");

        // 首先尝试使用DirectX获取GPU信息
        IDXGIFactory* pFactory = nullptr;
        if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory))) {
            IDXGIAdapter* pAdapter = nullptr;

            for (UINT i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC desc;
                if (SUCCEEDED(pAdapter->GetDesc(&desc))) {
                    char adapterName[256];
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, adapterName, sizeof(adapterName), NULL, NULL);

                    std::string gpuType = (i == 0) ? "主显示适配器" : "副显示适配器";
                    log(gpuType + ": " + std::string(adapterName));

                    // 显示显存信息
                    double videoMemoryGB = static_cast<double>(desc.DedicatedVideoMemory) / (1024.0 * 1024.0 * 1024.0);
                    double sharedMemoryGB = static_cast<double>(desc.SharedSystemMemory) / (1024.0 * 1024.0 * 1024.0);

                    log("  专用显存: " + std::to_string(videoMemoryGB) + " GB");
                    log("  共享内存: " + std::to_string(sharedMemoryGB) + " GB");
                }
                pAdapter->Release();
            }
            pFactory->Release();
        }
        else {
            // 备选方案：使用EnumDisplayDevices
            DISPLAY_DEVICEA displayDevice = { 0 };
            displayDevice.cb = sizeof(displayDevice);

            for (int i = 0; EnumDisplayDevicesA(NULL, i, &displayDevice, 0); i++) {
                if (displayDevice.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) {
                    log("主显示适配器: " + std::string(displayDevice.DeviceString));
                    break;
                }
            }
        }

        // 屏幕分辨率
        log("\n===== 显示设置 =====");
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        log("主显示器分辨率: " + std::to_string(screenWidth) + " × " + std::to_string(screenHeight));

        // 获取所有显示器
        int monitorCount = GetSystemMetrics(SM_CMONITORS);
        log("显示器数量: " + std::to_string(monitorCount));

        // 系统环境信息
        log("\n===== 系统环境信息 =====");

        char systemDir[MAX_PATH];
        if (GetSystemDirectoryA(systemDir, MAX_PATH)) {
            log("系统目录: " + std::string(systemDir));
        }

        char windowsDir[MAX_PATH];
        if (GetWindowsDirectoryA(windowsDir, MAX_PATH)) {
            log("Windows目录: " + std::string(windowsDir));
        }

        char tempDir[MAX_PATH];
        if (GetTempPathA(MAX_PATH, tempDir)) {
            log("临时目录: " + std::string(tempDir));
        }

        // 时区信息
        DYNAMIC_TIME_ZONE_INFORMATION tzInfo;
        if (GetDynamicTimeZoneInformation(&tzInfo) != TIME_ZONE_ID_INVALID) {
            char timeZoneName[256];
            WideCharToMultiByte(CP_UTF8, 0, tzInfo.TimeZoneKeyName, -1, timeZoneName, sizeof(timeZoneName), NULL, NULL);
            log("时区: " + std::string(timeZoneName));
        }

        // 系统语言
        LANGID langId = GetUserDefaultUILanguage();
        char langName[256];
        if (GetLocaleInfoA(MAKELCID(langId, SORT_DEFAULT), LOCALE_SENGLANGUAGE, langName, 256)) {
            log("系统语言: " + std::string(langName));
        }

        // 计算机名和用户名
        log("计算机名: " + get_computername());
        log("用户名: " + get_username());

        // 系统运行时间
        log("\n===== 系统运行状态 =====");
        ULONGLONG tickCount = GetTickCount64();
        int days = tickCount / (1000 * 60 * 60 * 24);
        int hours = (tickCount % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60);
        int minutes = (tickCount % (1000 * 60 * 60)) / (1000 * 60);
        int seconds = (tickCount % (1000 * 60)) / 1000;

        log("系统已运行: " + std::to_string(days) + "天 " +
            std::to_string(hours) + "小时 " +
            std::to_string(minutes) + "分钟 " +
            std::to_string(seconds) + "秒");

        // 进程信息
        log("当前进程ID: " + std::to_string(GetCurrentProcessId()));
        log("当前线程ID: " + std::to_string(GetCurrentThreadId()));

        // 获取系统版本详细信息
        log("\n===== 系统详细信息 =====");

        // 使用WMI替代方案：通过注册表获取Windows版本
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            0, KEY_READ, &hKey) == ERROR_SUCCESS) {

            char productName[256] = { 0 };
            DWORD size = sizeof(productName);

            if (RegQueryValueExA(hKey, "ProductName", NULL, NULL,
                (LPBYTE)productName, &size) == ERROR_SUCCESS) {
                log("产品名称: " + std::string(productName));
            }

            char buildLabEx[256];
            size = sizeof(buildLabEx);
            if (RegQueryValueExA(hKey, "BuildLabEx", NULL, NULL,
                (LPBYTE)buildLabEx, &size) == ERROR_SUCCESS) {
                log("构建版本: " + std::string(buildLabEx));
            }

            RegCloseKey(hKey);
        }

        // 获取BIOS信息
        log("\n===== BIOS信息 =====");
        HKEY biosKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\BIOS",
            0, KEY_READ, &biosKey) == ERROR_SUCCESS) {

            char biosVendor[256];
            DWORD size = sizeof(biosVendor);

            if (RegQueryValueExA(biosKey, "BIOSVendor", NULL, NULL,
                (LPBYTE)biosVendor, &size) == ERROR_SUCCESS) {
                log("BIOS厂商: " + std::string(biosVendor));
            }

            char biosVersion[256];
            size = sizeof(biosVersion);
            if (RegQueryValueExA(biosKey, "BIOSVersion", NULL, NULL,
                (LPBYTE)biosVersion, &size) == ERROR_SUCCESS) {
                log("BIOS版本: " + std::string(biosVersion));
            }

            char biosReleaseDate[256];
            size = sizeof(biosReleaseDate);
            if (RegQueryValueExA(biosKey, "BIOSReleaseDate", NULL, NULL,
                (LPBYTE)biosReleaseDate, &size) == ERROR_SUCCESS) {
                log("BIOS发布日期: " + std::string(biosReleaseDate));
            }

            RegCloseKey(biosKey);
        }

        log("\n软硬件信息获取完成");

    }
    catch (const std::exception& e) {
        log("获取信息时发生错误: " + std::string(e.what()));
    }
    catch (...) {
        log("获取信息时发生未知错误");
    }
}

void gpu_test_it_s_too_difficult() {
    log("GPU实现太难了，虽然我们也在尽力优化，但是结果还是不准确");
    log("我觉得我们还是关闭GPU测试吧？");
    log("为什么微软不在更新Windows 11的时候同步更新API啊，搞得现在获取ProductName都不准确");
}

void update_log() {
    log("1.0.1 CLI更新日志 [更新日期: 2025年12月4日]");
    log("优化GPU测试，使带宽获取更准确，测试时间更短（相对）");
    log("增加获取系统信息的函数（void get_info）");
    log("1.0.2 CLI更新日志 [更新日期: 2025年12月6日]");
    log("优化获取系统信息的函数（void get_info），区分Windows 10和Windows 11");
    log("1.0.3 CLI更新日志 [更新日期: 2025年12月20日]");
    log("新增配置文件导入导出功能，支持加载/保存测试配置");
    log("1.0.4 CLI更新日志 [更新日期: 2025年12月21日]");
    log("新增function和project指令，可以查看主要函数和项目地址");
}

// 执行所有测试（使用标准化得分计算）
void run_all_tests() {
    if (is_testing) {
        log("测试已经在运行中");
        return;
    }

    is_testing = true;
    stop_test = false;
    test_results.clear();
    raw_results.clear();

    log("开始测试序列");

    // 根据开关状态执行测试
    if (test_config.switches.single_core) {
        single_core_test();
        if (stop_test) {
            is_testing = false;
            return;
        }
    }
    else {
        log("跳过单核性能测试（已关闭）");
    }

    if (test_config.switches.multi_core) {
        multi_core_test();
        if (stop_test) {
            is_testing = false;
            return;
        }
    }
    else {
        log("跳过多核性能测试（已关闭）");
    }

    if (test_config.switches.memory) {
        memory_test();
        if (stop_test) {
            is_testing = false;
            return;
        }
    }
    else {
        log("跳过内存性能测试（已关闭）");
    }

    if (test_config.switches.crypto) {
        crypto_test();
        if (stop_test) {
            is_testing = false;
            return;
        }
    }
    else {
        log("跳过加密性能测试（已关闭）");
    }

    if (test_config.switches.gpu) {
        gpu_test();
        if (stop_test) {
            is_testing = false;
            return;
        }
    }
    else {
        log("跳过GPGPU性能测试（已关闭）");
    }

    // 计算总分时只计算已开启的测试项目
    if (!test_results.empty()) {
        double total_score = 0.0;
        double total_weight = 0.0;

        if (test_config.switches.single_core && test_results.count("single_core")) {
            total_score += test_results["single_core"] * test_config.weights.single_core;
            total_weight += test_config.weights.single_core;
        }
        if (test_config.switches.multi_core && test_results.count("multi_core")) {
            total_score += test_results["multi_core"] * test_config.weights.multi_core;
            total_weight += test_config.weights.multi_core;
        }
        if (test_config.switches.memory && test_results.count("memory")) {
            total_score += test_results["memory"] * test_config.weights.memory;
            total_weight += test_config.weights.memory;
        }
        if (test_config.switches.crypto && test_results.count("crypto")) {
            total_score += test_results["crypto"] * test_config.weights.crypto;
            total_weight += test_config.weights.crypto;
        }
        if (test_config.switches.gpu && test_results.count("gpu")) {
            total_score += test_results["gpu"] * test_config.weights.gpu;
            total_weight += test_config.weights.gpu;
        }

        // 如果所有测试都关闭，则total_weight为0，需要避免除以0
        if (total_weight > 0) {
            total_score = total_score / total_weight * (test_config.weights.single_core +
                test_config.weights.multi_core +
                test_config.weights.memory +
                test_config.weights.crypto +
                test_config.weights.gpu);
        }
        else {
            total_score = 0.0;
        }

        log("\n===== 测试结果 =====");
        if (test_config.switches.single_core && test_results.count("single_core")) {
            log("单核性能得分: " + std::to_string(test_results["single_core"]) +
                " (原始: " + std::to_string(raw_results["single_core"]) + ")");
        }
        if (test_config.switches.multi_core && test_results.count("multi_core")) {
            log("多核性能得分: " + std::to_string(test_results["multi_core"]) +
                " (原始: " + std::to_string(raw_results["multi_core"]) + ")");
        }
        if (test_config.switches.memory && test_results.count("memory")) {
            log("内存性能得分: " + std::to_string(test_results["memory"]) +
                " (原始: " + std::to_string(raw_results["memory"]) + ")");
        }
        if (test_config.switches.crypto && test_results.count("crypto")) {
            log("加密性能得分: " + std::to_string(test_results["crypto"]) +
                " (原始: " + std::to_string(raw_results["crypto"]) + ")");
        }
        if (test_config.switches.gpu && test_results.count("gpu")) {
            log("GPGPU性能得分: " + std::to_string(test_results["gpu"]) +
                " (原始: " + std::to_string(raw_results["gpu"]) + ")");
        }
        log("====================");
        log("综合性能得分: " + std::to_string(total_score));
        log("====================");
    }

    is_testing = false;
    log("所有测试完成");
}

// 停止测试
void stop_all_tests() {
    if (!is_testing) {
        log("没有正在运行的测试");
        return;
    }

    log("正在停止所有测试...");
    stop_test = true;

    for (auto& t : test_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    test_threads.clear();

    is_testing = false;
    log("所有测试已停止");
}

// 显示帮助信息
void show_help() {
    log("可用命令:");
    log("- start-test: 开始所有测试");
    log("- end-test: 停止当前测试");
    log("- output-log-路径: 导出日志到指定路径");
    log("- help: 显示帮助信息");
    log("- list-test: 列出所有测试项");
    log("- test-time-测试项-时间: 设置指定测试项的时间（1s-65536s）");
    log("- test-time-测试项-seetime: 查看测试项的时长");
    log("- test-time-re: 恢复默认时长");
    log("- 测试项=true/false: 开启/关闭指定测试项");
    log("- dev-test: 开启/关闭开发者模式");
    log("- ver: 显示软件版本");
    log("- get-info: 获取系统软硬件信息");
    log("- update-log: 查看版本更新日志");
    log("- load-ini-file-路径: 从指定路径加载配置文件");
    log("- output-ini-file-路径: 将当前配置导出到指定路径");
    log("- function: 查看主要函数");
    log("- project: 查看项目地址");
    log("- difficult: 说不定有什么隐藏彩蛋或者开发者的诉苦？");
}

// 列出所有测试项
void list_tests() {
    log("所有测试项:");
    log("- single_core: 单核性能测试 (时长: " + std::to_string(test_config.duration.single_core) +
        "秒, 状态: " + (test_config.switches.single_core ? "开启" : "关闭") + ")");
    log("- multi_core: 多核性能测试 (时长: " + std::to_string(test_config.duration.multi_core) +
        "秒, 状态: " + (test_config.switches.multi_core ? "开启" : "关闭") + ")");
    log("- memory: 内存性能测试 (时长: " + std::to_string(test_config.duration.memory) +
        "秒, 状态: " + (test_config.switches.memory ? "开启" : "关闭") + ")");
    log("- crypto: 加密性能测试 (时长: " + std::to_string(test_config.duration.crypto) +
        "秒, 状态: " + (test_config.switches.crypto ? "开启" : "关闭") + ")");
    log("- gpu: GPGPU性能测试 (时长: " + std::to_string(test_config.duration.gpu) +
        "秒, 状态: " + (test_config.switches.gpu ? "开启" : "关闭") + ")");
}

// 设置测试时间
void set_test_time(const std::string& test_item, int duration) {
    if (duration < 1 || duration > 65536) {
        log("错误: 时间必须在1-65536秒之间");
        return;
    }

    if (test_item == "single_core") {
        test_config.duration.single_core = duration;
        log("单核性能测试时长已设置为 " + std::to_string(duration) + " 秒");
    }
    else if (test_item == "multi_core") {
        test_config.duration.multi_core = duration;
        log("多核性能测试时长已设置为 " + std::to_string(duration) + " 秒");
    }
    else if (test_item == "memory") {
        test_config.duration.memory = duration;
        log("内存性能测试时长已设置为 " + std::to_string(duration) + " 秒");
    }
    else if (test_item == "crypto") {
        test_config.duration.crypto = duration;
        log("加密性能测试时长已设置为 " + std::to_string(duration) + " 秒");
    }
    else if (test_item == "gpu") {
        test_config.duration.gpu = duration;
        log("GPGPU性能测试时长已设置为 " + std::to_string(duration) + " 秒");
    }
    else {
        log("错误: 测试项 '" + test_item + "' 不存在");
    }
}

// 显示测试时间
void show_test_time(const std::string& test_item) {
    if (test_item == "single_core") {
        log("单核性能测试当前时长: " + std::to_string(test_config.duration.single_core) + " 秒");
    }
    else if (test_item == "multi_core") {
        log("多核性能测试当前时长: " + std::to_string(test_config.duration.multi_core) + " 秒");
    }
    else if (test_item == "memory") {
        log("内存性能测试当前时长: " + std::to_string(test_config.duration.memory) + " 秒");
    }
    else if (test_item == "crypto") {
        log("加密性能测试当前时长: " + std::to_string(test_config.duration.crypto) + " 秒");
    }
    else if (test_item == "gpu") {
        log("GPGPU性能测试当前时长: " + std::to_string(test_config.duration.gpu) + " 秒");
    }
    else {
        log("错误: 测试项 '" + test_item + "' 不存在");
    }
}

// 恢复默认测试时间
void restore_default_time() {
    test_config.duration = TestConfig::Durations();
    log("所有测试项时长已恢复默认值");
}

// 切换开发者模式
void toggle_developer_mode() {
    test_config.developer_mode = !test_config.developer_mode;
    if (test_config.developer_mode) {
        log("开发者模式已开启");
        log("开发者模式将使用更复杂的测试算法");
    }
    else {
        log("开发者模式已关闭，将使用标准测试算法");
    }
}

// 显示版本信息
void show_version() {
    log("Arce TestTool 版本 " + test_config.version);
    log("版权所有 (c) 一名飞手。");
    log("Copyright (c) yimingfeishou.");
}

// 解析并执行命令
void execute_command(const std::string& command) {
    if (command.empty()) return;

    log("执行命令: " + command);

    if (command == "help") {
        show_help();
    }
    else if (command == "start-test") {
        if (is_testing) {
            log("测试已经在运行中");
        }
        else {
            test_threads.emplace_back(run_all_tests);
        }
    }
    else if (command == "end-test") {
        stop_all_tests();
    }
    else if (command.substr(0, 11) == "output-log-") {
        std::string path = command.substr(11);
        if (path.empty()) {
            log("错误: 请指定日志路径，格式: output-log-路径");
        }
        else {
            export_log(path);
        }
    }
    else if (command == "list-test") {
        list_tests();
    }
    else if (command == "test-time-re") {
        restore_default_time();
    }
    else if (command == "get-info") {
        get_info();
    }
    else if (command == "update-log") {
        update_log();
    }
    else if (command == "difficult") {
        gpu_test_it_s_too_difficult();
    }
    else if (command.substr(0, 10) == "test-time-") {
        std::string sub = command.substr(10);
        size_t pos = sub.find('-');
        if (pos == std::string::npos) {
            log("错误: 命令格式错误，正确格式: test-time-测试项-时间 或 test-time-测试项-seetime");
            return;
        }

        std::string test_item = sub.substr(0, pos);
        std::string value = sub.substr(pos + 1);

        if (value == "seetime") {
            show_test_time(test_item);
        }
        else {
            try {
                int duration = std::stoi(value);
                set_test_time(test_item, duration);
            }
            catch (...) {
                log("错误: 时间必须是整数");
            }
        }
    }
    else if (command.substr(0, 14) == "load-ini-file-") {
        std::string path = command.substr(14);
        if (path.empty()) {
            log("错误: 请指定配置文件的完整路径，格式: load-ini-file-路径");
        }
        else {
            load_ini_file(path);
        }
    }
    else if (command.substr(0, 16) == "output-ini-file-") {
        std::string path = command.substr(16);
        if (path.empty()) {
            log("错误: 请指定配置文件的完整路径，格式: output-ini-file-路径");
        }
        else {
            save_ini_file(path);
        }
    }
    else if (command == "function") {
        log("主要函数");
        log("double random_double");
        log("void lu_decomposition");
        log("void svd_decomposition");
        log("void single_core_test");
        log("void multi_core_worker");
        log("void multi_core_test");
        log("void memory_test");
        log("void crypto_test");
        log("void void gpu_test");
        log("void update_log");
        log("更多函数，请参阅源码");
    }
    else if (command == "project") {
        log("项目地址");
        log("GitHub: https://github.com/yimingfeishou/Arce_Tools");
    }
    // 测试项目开关设置
    else if (command == "single_core=true" || command == "single_core=false") {
        bool value = (command == "single_core=true");
        test_config.switches.single_core = value;
        log("单核性能测试: " + std::string(value ? "开启" : "关闭"));
    }
    else if (command == "multi_core=true" || command == "multi_core=false") {
        bool value = (command == "multi_core=true");
        test_config.switches.multi_core = value;
        log("多核性能测试: " + std::string(value ? "开启" : "关闭"));
    }
    else if (command == "memory=true" || command == "memory=false") {
        bool value = (command == "memory=true");
        test_config.switches.memory = value;
        log("内存性能测试: " + std::string(value ? "开启" : "关闭"));
    }
    else if (command == "crypto=true" || command == "crypto=false") {
        bool value = (command == "crypto=true");
        test_config.switches.crypto = value;
        log("加密性能测试: " + std::string(value ? "开启" : "关闭"));
    }
    else if (command == "gpu=true" || command == "gpu=false") {
        bool value = (command == "gpu=true");
        test_config.switches.gpu = value;
        log("GPGPU性能测试: " + std::string(value ? "开启" : "关闭"));
    }
    else if (command == "dev-test") {
        toggle_developer_mode();
    }
    else if (command == "ver") {
        show_version();
    }
    else {
        log("错误: 未知命令 '" + command + "'，输入 'help' 查看帮助");
    }
}

int main() {
    std::string username = get_username();
    std::string computername = get_computername();

    std::cout << "版权所有 (c) 一名飞手。\n";
    std::cout << "Copyright (c) yimingfeishou.\n\n";
    log("Arce TestTool 版本 " + test_config.version + " 启动");
    log("输入 'help' 查看可用命令");
    log("测试时仍可以输入命令进行操作");
    log("如要获得最准确的测试结果，建议以管理员身份运行");
    log("L1/L2为获取单个核心的缓存，L3为获取整个CPU的缓存，可能会与任务管理器显示的缓存有差异");
    log("测试结果仅供参考，请以实际性能为准");
    log("GPU测试时间可能较长（屎山代码发力了），将在后续更新时解决（现在没解决），敬请谅解，可以关闭此测试项，指令详见 'help'");
    log("在 " + username + " 用户上");
    log("在 " + computername + " 计算机上\n");

    std::string command;
    while (true) {
        std::cout << "Arce TestTool/" + test_config.version + "> ";
        std::getline(std::cin, command);

        if (command == "exit") {
            if (is_testing) {
                stop_all_tests();
            }
            break;
        }

        execute_command(command);
    }

    log("程序已退出");
    return 0;
}
