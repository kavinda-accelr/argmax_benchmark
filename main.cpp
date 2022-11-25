#include <iostream>
#include <future>
#include <array>
#include "Thread_Pool.hpp"
#include "Timer.hpp"

#define NUM_THREADS 4

class ThreadPool_Wait
{
public:
    ThreadPool_Wait(const unsigned int num_threads=0): m_task_count(0), m_join(false)
    {
        m_num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
        for(unsigned int i=0; i<m_num_threads; i++)
        {
            m_threads.emplace_back(std::thread(ThreadPool_Wait::thread_work, this));
        }
    }

    void assign(std::function<void()> work)
    {
        m_queue_mutex.lock();
        m_work_queue.push(work);
        m_queue_mutex.unlock();
        m_cv.notify_one();
    }

    void join()
    {
        m_join = true;
        m_cv.notify_all();
        for(auto& t : m_threads)
        {
            if(t.joinable()) t.join();
        }
    }

    void wait_until(const unsigned int task_count)
    {
        // wait only if task count not completed and join not called
        while (m_task_count < task_count && !m_join) std::this_thread::yield();
        m_task_count = 0;
    }

    unsigned int get_num_threads() const
    {
        return m_num_threads;
    }

    ~ThreadPool_Wait()
    {
        join();
    }
private:
    static void thread_work(ThreadPool_Wait *threadPool)
    {
        std::function<void()> work;
        bool work_assigned = false;
        std::unique_lock<std::mutex> cv_lck(threadPool->m_cv_mutex, std::defer_lock);
        std::unique_lock<std::mutex> queue_lck(threadPool->m_queue_mutex, std::defer_lock);
        //break the loop if only join is called and queue is empty 
        while (!(threadPool->m_join && threadPool->m_work_queue.empty())) 
        {
            cv_lck.lock();
            //call wait only if join is not called and queue is empty 
            if(!threadPool->m_join && threadPool->m_work_queue.empty())
            {
                //wait only if join is not called and queue is empty
                threadPool->m_cv.wait(cv_lck, [&threadPool]()
                {return !(!threadPool->m_join && threadPool->m_work_queue.empty());}); 
            }
            cv_lck.unlock();
 
            queue_lck.lock();
            if(!threadPool->m_work_queue.empty())
            {
                work = threadPool->m_work_queue.front();
                threadPool->m_work_queue.pop();
                work_assigned = true;
            }
            queue_lck.unlock();
            if(work_assigned) 
            {
                work();
                threadPool->m_task_count++;
                work_assigned = false;
            }
        }
    }
    std::vector<std::thread> m_threads;
    std::mutex m_queue_mutex;
    std::mutex m_cv_mutex;
    std::condition_variable m_cv;
    std::atomic_bool m_join;
    unsigned int m_num_threads;
    std::atomic_uint16_t m_task_count;
    std::queue<std::function<void()>> m_work_queue;
};

template<typename T>
inline unsigned int argmax(const T* const arr_ptr, unsigned const int size)
{
    const T* max_val_ptr = arr_ptr;
    for(unsigned int i = 1; i<size; i++)
    {
        max_val_ptr = arr_ptr[i] > *max_val_ptr ? (arr_ptr + i) : max_val_ptr;
    }
    return (unsigned int)(max_val_ptr - arr_ptr);
}

template <typename T>
inline void argmax_tensor(const T* tensor_ptr, T* const mat_ptr, const unsigned int num_filters, const unsigned int mat_size)
{
    for(unsigned int i=0; i<mat_size; i++)
    {
        mat_ptr[i] = (T)argmax(tensor_ptr, num_filters);
        tensor_ptr += num_filters;
    }
}

template <typename T>
void argmax_tensor_mt_thread_pool(
    const T* tensor_ptr, 
    T* const mat_ptr, 
    const unsigned int num_filters, 
    const unsigned int mat_size, 
    obj_detect::Thread_Pool& thread_pool)
{
    const unsigned int num_threads = thread_pool.get_num_threads();
    const unsigned int work_per_thread = mat_size/num_threads;
    const unsigned int work_left = mat_size%num_threads;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    for(unsigned int i=0; i<num_threads; i++)
    {
        work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
        thread_pool.assign([&, total_work_count, work_count](){
            const T* c_tensor_ptr = tensor_ptr + num_filters*total_work_count;
            T* const c_mat_ptr = mat_ptr + total_work_count;
            const T* max_val_ptr = NULL;
            unsigned int i, j;
            for(i=0; i<work_count; i++)
            {
                max_val_ptr = c_tensor_ptr;
                for(j = 1; j<num_filters; j++)
                {
                    max_val_ptr = c_tensor_ptr[j] > *max_val_ptr ? (c_tensor_ptr + j) : max_val_ptr;
                }
                c_mat_ptr[i] = (unsigned int)(max_val_ptr - c_tensor_ptr);   
                c_tensor_ptr += num_filters;
            }
        });
        total_work_count += work_count;
    }
    thread_pool.wait_until(num_threads);
}

template <typename T>
void argmax_tensor_mt_thread_pool_wait(
    const T* tensor_ptr, 
    T* const mat_ptr, 
    const unsigned int num_filters, 
    const unsigned int mat_size, 
    ThreadPool_Wait& thread_pool)
{
    const unsigned int num_threads = thread_pool.get_num_threads();
    const unsigned int work_per_thread = mat_size/num_threads;
    const unsigned int work_left = mat_size%num_threads;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    for(unsigned int i=0; i<num_threads; i++)
    {
        work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
        thread_pool.assign([&, total_work_count, work_count](){
            const T* c_tensor_ptr = tensor_ptr + num_filters*total_work_count;
            T* const c_mat_ptr = mat_ptr + total_work_count;
            const T* max_val_ptr = NULL;
            unsigned int i, j;
            for(i=0; i<work_count; i++)
            {
                max_val_ptr = c_tensor_ptr;
                for(j = 1; j<num_filters; j++)
                {
                    max_val_ptr = c_tensor_ptr[j] > *max_val_ptr ? (c_tensor_ptr + j) : max_val_ptr;
                }
                c_mat_ptr[i] = (unsigned int)(max_val_ptr - c_tensor_ptr);   
                c_tensor_ptr += num_filters;
            }
        });
        total_work_count += work_count;
    }
    thread_pool.wait_until(num_threads);
}


template <typename T>
void argmax_tensor_mt_async(
    const T* tensor_ptr, 
    T* const mat_ptr, 
    const unsigned int num_filters, 
    const unsigned int mat_size,
    const unsigned int num_threads)
{
    std::vector<std::future<void>> fs(num_threads);
    const unsigned int work_per_thread = mat_size/num_threads;
    const unsigned int work_left = mat_size%num_threads;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    for(unsigned int i=0; i<num_threads; i++)
    {
        work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
        fs[i] = std::async(std::launch::async, [&, total_work_count, work_count](){
            const T* c_tensor_ptr = tensor_ptr + num_filters*total_work_count;
            T* const c_mat_ptr = mat_ptr + total_work_count;
            const T* max_val_ptr = NULL;
            unsigned int i, j;
            for(i=0; i<work_count; i++)
            {
                max_val_ptr = c_tensor_ptr;
                for(j = 1; j<num_filters; j++)
                {
                    max_val_ptr = c_tensor_ptr[j] > *max_val_ptr ? (c_tensor_ptr + j) : max_val_ptr;
                }
                c_mat_ptr[i] = (unsigned int)(max_val_ptr - c_tensor_ptr);   
                c_tensor_ptr += num_filters;
            }
        });
        total_work_count += work_count;
    }
    for(auto& f : fs)
    {
        f.wait();
    }
}


template <typename T>
void comp_vec(const std::vector<T> vec_1, const std::vector<T> vec_2)
{
    if(vec_1.size() != vec_2.size())
    {
        std::cerr<<"size mismatch\n";
        return;
    }

    for(int i=0; i<vec_1.size(); i++)
    {
        if(vec_1[i] != vec_2[i])
        {
            std::cerr<<"value mismatch : "<< (int)vec_1[i]<< " != " << (int)vec_2[i] <<std::endl;
            return;
        }
    }
}

void fill_vec(std::vector<int8_t>& vec)
{
    for(auto& item : vec)
    {
        item = rand()%256 - 128;
    }
}

void test_argmax_mt()
{
    for(unsigned int i=0; i<10; i++)
    {
        srand((unsigned int)time(NULL)+i*10);
        const unsigned int num_theads = rand()%15 + 1;
        const unsigned int num_columns = rand()%200 + 1;
        const unsigned int num_rows = rand()%200 + 1;;
        const unsigned int num_filters = rand()%200 + 1;;

        const unsigned int tensor_size = num_columns*num_rows*num_filters;
        const unsigned int mat_size = num_columns*num_rows;

        obj_detect::Thread_Pool thread_pool(num_theads);
        ThreadPool_Wait thread_pool_wait(num_theads);
        std::vector<int8_t> tensor(tensor_size);
        std::vector<int8_t> mat_1(mat_size);
        std::vector<int8_t> mat_2(mat_size);
        std::vector<int8_t> mat_3(mat_size);
        std::vector<int8_t> mat_4(mat_size);

        fill_vec(tensor);

        argmax_tensor(tensor.data(), mat_1.data(), num_filters, mat_size);
        argmax_tensor_mt_thread_pool(tensor.data(), mat_2.data(), num_filters, mat_size, thread_pool);
        argmax_tensor_mt_thread_pool_wait(tensor.data(), mat_3.data(), num_filters, mat_size, thread_pool_wait);
        argmax_tensor_mt_async(tensor.data(), mat_4.data(), num_filters, mat_size, num_theads);

        comp_vec(mat_1, mat_2);
        comp_vec(mat_1, mat_3);
        comp_vec(mat_1, mat_4);

        std::cout<<"I : "<< i<<" | ";
        // std::cout<<"T : "<< num_theads<<" | ";
        // std::cout<<"C : "<< num_columns<<" | ";
        // std::cout<<"R : "<< num_rows<<" | ";
        // std::cout<<"F : "<< num_filters<<std::endl;
    }
    std::cout<<std::endl;
}

// Argmax MT-TP
void argmax_mt_benchmark_tp(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand((unsigned int)time(NULL));
    const unsigned int num_theads = NUM_THREADS;
    obj_detect::Thread_Pool thread_pool(num_theads);
    
    for(unsigned int c=0; c<cycles; c++)
    {
        fill_vec(tensor);
        Timer::Get().start("Argmax MT-TP-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        argmax_tensor_mt_thread_pool(tensor.data(), mat.data(), num_filters, mat_size, thread_pool);
        Timer::Get().stop();
    }
}

// Argmax MT-TPW
void argmax_mt_benchmark_tpw(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand((unsigned int)time(NULL));
    const unsigned int num_theads = NUM_THREADS;
    ThreadPool_Wait thread_pool(num_theads);
    
    for(unsigned int c=0; c<cycles; c++)
    {
        fill_vec(tensor);
        Timer::Get().start("Argmax MT-TPW-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        argmax_tensor_mt_thread_pool_wait(tensor.data(), mat.data(), num_filters, mat_size, thread_pool);
        Timer::Get().stop();
    }
}

//Argmax MT-AS
void argmax_mt_benchmark_as(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand((unsigned int)time(NULL));
    const unsigned int num_theads = NUM_THREADS;
    obj_detect::Thread_Pool thread_pool(num_theads);
    
    for(unsigned int c=0; c<cycles; c++)
    {
        fill_vec(tensor);
        Timer::Get().start("Argmax MT-AS-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        argmax_tensor_mt_async(tensor.data(), mat.data(), num_filters, mat_size, num_theads);
        Timer::Get().stop();
    }
}

// Argmax MT
void argmax_mt_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles
)
{
    const unsigned int num_theads = NUM_THREADS;
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand((unsigned int)time(NULL));
    obj_detect::Thread_Pool thread_pool(num_theads);
    const unsigned int work_per_thread = mat_size/num_theads;
    const unsigned int work_left = mat_size%num_theads;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    
    for(unsigned int c=0; c<cycles; c++)
    {
        fill_vec(tensor);

        Timer::Get().start("Argmax MT-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        total_work_count = 0;
        for(unsigned int i=0; i<num_theads; i++)
        {
            work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
            thread_pool.assign([&, total_work_count, work_count](){
                argmax_tensor(
                    tensor.data() + num_filters*total_work_count, 
                    mat.data() + total_work_count, 
                    num_filters, 
                    work_count);
            });
            total_work_count += work_count;
        }

        thread_pool.wait_until(num_theads);
        Timer::Get().stop();
    }
}

// Argmax ST
void argmax_st_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles
)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand((unsigned int)time(NULL));
    for(unsigned int c=0; c<cycles; c++)
    {
        fill_vec(tensor);

        Timer::Get().start("Argmax ST-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        argmax_tensor(tensor.data(), mat.data(), num_filters, mat_size);
        Timer::Get().stop();
    }
}

void sim()
{
    const unsigned int cycles = 10;
    const unsigned int rows = 224;
    const unsigned int columns = 224;
    const unsigned int filters = 224;

    test_argmax_mt();
    argmax_mt_benchmark_tp(rows, columns, filters, cycles);
    argmax_mt_benchmark_tpw(rows, columns, filters, cycles);
    argmax_mt_benchmark_as(rows, columns, filters, cycles);
    argmax_mt_benchmark(rows, columns, filters, cycles);
    argmax_st_benchmark(rows, columns, filters, cycles);

    Timer::Get().print_duration();
}

int main()
{
    const unsigned int cycles = 100;
    const unsigned int rows = 5000;
    const unsigned int columns = 5000;
    const unsigned int filters = 21;

    argmax_st_benchmark(rows, columns, filters, cycles);
    argmax_mt_benchmark_tp(rows, columns, filters, cycles);
    argmax_mt_benchmark_tpw(rows, columns, filters, cycles);

    Timer::Get().print_duration();

    return 0;
}