#pragma once

#include <map>
#include <string>
#include <iostream>
#include <iomanip>

#if __linux__ == 1
struct Time_data
{
    double t_cpu_time_used = 0;
    double t_process_cpu = 0;
    double t_monotonic = 0;
    double t_monotonic_raw = 0;
    double t_real = 0;
    unsigned int cycles = 0;
};
#else
struct Time_data
{
    double time = 0.0;
    unsigned int cycles = 0;
};
#endif
class Base_Timer
{
public:
    Base_Timer() {}
    virtual void start(std::string map_name) = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual void print_duration() = 0;
    virtual ~Base_Timer() {}
protected:
    std::string m_map_name;
    std::map<std::string, Time_data> m_time_data;
};
#if __linux__ == 1
#include <ctime>
class Timer : private Base_Timer
{
public:
    Timer(const Timer&) = delete;

    static Timer& Get()
    {
        static Timer instance;
        return instance;
    }

    void start(std::string map_name) override
    {
        m_map_name = map_name;
        if (m_time_data.find(m_map_name) == m_time_data.end())
        {
            m_time_data[m_map_name].t_cpu_time_used = 0.0;
            m_time_data[m_map_name].t_process_cpu = 0.0;
            m_time_data[m_map_name].t_monotonic = 0.0;
            m_time_data[m_map_name].t_monotonic_raw = 0.0;
            m_time_data[m_map_name].t_real = 0.0;
            m_time_data[m_map_name].cycles = 0;
        }

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t_process_cpu_1);
        clock_gettime(CLOCK_MONOTONIC, &t_monotonic_1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_monotonic_raw_1);
        clock_gettime(CLOCK_REALTIME, &t_real_1);
        t1 = clock();
    }

    void stop() override
    {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t_process_cpu_2);
        clock_gettime(CLOCK_MONOTONIC, &t_monotonic_2);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_monotonic_raw_2);
        clock_gettime(CLOCK_REALTIME, &t_real_2);
        t2 = clock();

        m_time_data.at(m_map_name).t_cpu_time_used += 1000.0 * (t2 - t1) / CLOCKS_PER_SEC;

        m_time_data.at(m_map_name).t_process_cpu += (1000.0 * t_process_cpu_2.tv_sec + 1e-6 * t_process_cpu_2.tv_nsec) -
            (1000.0 * t_process_cpu_1.tv_sec + 1e-6 * t_process_cpu_1.tv_nsec);
        m_time_data.at(m_map_name).t_monotonic += (1000.0 * t_monotonic_2.tv_sec + 1e-6 * t_monotonic_2.tv_nsec) -
            (1000.0 * t_monotonic_1.tv_sec + 1e-6 * t_monotonic_1.tv_nsec);
        m_time_data.at(m_map_name).t_monotonic_raw += (1000.0 * t_monotonic_raw_2.tv_sec + 1e-6 * t_monotonic_raw_2.tv_nsec) -
            (1000.0 * t_monotonic_raw_1.tv_sec + 1e-6 * t_monotonic_raw_1.tv_nsec);
        m_time_data.at(m_map_name).t_real += (1000.0 * t_real_2.tv_sec + 1e-6 * t_real_2.tv_nsec) -
            (1000.0 * t_real_1.tv_sec + 1e-6 * t_real_1.tv_nsec);
        m_time_data.at(m_map_name).cycles += 1;
    }
    void reset() override
    {
        m_time_data.clear();
    }
    void print_duration() override
    {

        std::cout << "Time in ms" << std::endl;
        std::cout
            << std::left << std::setw(25) << "Bolck name"
            << std::left << std::setw(20) << "CPU TIME USED"
            << std::left << std::setw(20) << "PROCESS_CPUTIME_ID"
            << std::left << std::setw(20) << "MONOTONIC"
            << std::left << std::setw(20) << "MONOTONIC_RAW"
            << std::left << std::setw(20) << "REALTIME"
            << std::left << std::setw(20) << "Cycles"
            << std::endl << std::endl;

        for (auto it = m_time_data.begin(); it != m_time_data.end(); it++) {
            std::string name = it->first;
            const unsigned int cycles = m_time_data.at(name).cycles;
            const double t_cpu_time_used = m_time_data.at(name).t_cpu_time_used;
            const double t_monotonic = m_time_data.at(name).t_monotonic;
            const double t_monotonic_raw = m_time_data.at(name).t_monotonic_raw;
            const double t_process_cpu = m_time_data.at(name).t_process_cpu;
            const double t_real = m_time_data.at(name).t_real;

            std::cout
                << std::left << std::setw(25) << name
                << std::left << std::setw(20) << t_cpu_time_used / cycles
                << std::left << std::setw(20) << t_process_cpu / cycles
                << std::left << std::setw(20) << t_monotonic / cycles
                << std::left << std::setw(20) << t_monotonic_raw / cycles
                << std::left << std::setw(20) << t_real / cycles
                << std::left << std::setw(20) << cycles
                << std::endl;
        }
    }
    ~Timer() {}
private:
    Timer() :Base_Timer() {}
    struct timespec t_process_cpu_1, t_monotonic_1, t_monotonic_raw_1, t_real_1;
    struct timespec t_process_cpu_2, t_monotonic_2, t_monotonic_raw_2, t_real_2;
    clock_t t1, t2;
};
#else
#include <chrono>
class Timer : private Base_Timer
{
public:
    Timer(const Timer&) = delete;

    static Timer& Get()
    {
        static Timer instance;
        return instance;
    }

    void start(std::string map_name) override
    {
        m_map_name = map_name;
        if (m_time_data.find(m_map_name) == m_time_data.end())
        {
            m_time_data[m_map_name].time = 0.0;
            m_time_data[m_map_name].cycles = 0;
        }
        m_StartTimePoint = std::chrono::high_resolution_clock::now();
    }
    void stop() override
    {
        auto entTimePoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(entTimePoint).time_since_epoch().count();

        auto duration = (end - start) * 0.001;
        m_time_data.at(m_map_name).time += duration;
        m_time_data.at(m_map_name).cycles += 1;
    }
    void reset() override
    {
        m_time_data.clear();
    }
    void print_duration() override
    {

        std::cout << std::left << std::setw(30) << "Bolck name" << std::left << std::setw(20) << "Time (ms)" << std::left << std::setw(10) << "Cycles" << std::endl;
        std::cout << std::left << std::setw(30) << "----------" << std::left << std::setw(20) << "---------" << std::left << std::setw(10) << "------" << std::endl;

        for (auto it = m_time_data.begin(); it != m_time_data.end(); it++) {
            std::string name = it->first;
            const unsigned int cycles = m_time_data.at(name).cycles;
            const double time = m_time_data.at(name).time;
            std::cout << std::left << std::setw(30) << name << std::left << std::setw(20) << time / cycles << std::left << std::setw(10) << cycles << std::endl;
        }
    }
    ~Timer() {}
private:
    Timer() :Base_Timer() {}
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
};
#endif