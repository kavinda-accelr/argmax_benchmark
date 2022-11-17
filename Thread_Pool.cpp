#include "Thread_Pool.hpp"

obj_detect::Thread_Pool::Thread_Pool(const unsigned int num_threads = 0) : _task_count(0), _join(false)
{
    _num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < _num_threads; i++)
    {
        _threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
    }
}

void obj_detect::Thread_Pool::assign(std::function<void()> work)
{
    std::unique_lock<std::mutex> queue_lck(_queue_mutex);
    _work_queue.push(work);
    queue_lck.unlock();
}

void obj_detect::Thread_Pool::join()
{
    _join = true;
    for (auto& t : _threads)
    {
        if (t.joinable()) t.join();
    }
}

unsigned int obj_detect::Thread_Pool::get_num_threads() const
{
    return _num_threads;
}

void obj_detect::Thread_Pool::wait_until(const unsigned int task_cout)
{
    // wait only if task count not completed and join not called
    while (_task_count < task_cout && !_join) std::this_thread::yield();
    _task_count = 0;
}

obj_detect::Thread_Pool::~Thread_Pool()
{
    join();
}

void obj_detect::Thread_Pool::thread_work(Thread_Pool* threadPool)
{
    std::function<void()> work;
    bool work_assigned = false;
    std::unique_lock<std::mutex> queue_lck(threadPool->_queue_mutex, std::defer_lock);
    while (!(threadPool->_join && threadPool->_work_queue.empty())) //break the loop if only join is called and queue is empty 
    {
        if (threadPool->_work_queue.empty()) std::this_thread::yield();
        else
        {
            queue_lck.lock();
            if (!threadPool->_work_queue.empty())
            {
                work = threadPool->_work_queue.front();
                threadPool->_work_queue.pop();
                work_assigned = true;
            }
            queue_lck.unlock();
            if (work_assigned)
            {
                work();
                threadPool->_task_count++;
                work_assigned = false;
            }
        }
    }
}