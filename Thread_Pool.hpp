#pragma once

#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace obj_detect
{
    class Thread_Pool
    {
    public:
        Thread_Pool(const unsigned int num_threads);

        void assign(std::function<void()> work);

        void join();

        unsigned int get_num_threads() const;

        void wait_until(const unsigned int task_cout);

        ~Thread_Pool();
    private:
        static void thread_work(Thread_Pool* threadPool);

        std::atomic_bool _join;
        std::mutex _queue_mutex;
        std::queue<std::function<void()>> _work_queue;
        unsigned int _num_threads;
        std::vector<std::thread> _threads;
        std::atomic_uint16_t _task_count;
    };
}