#include <iostream>
#include <future>
#include <array>
#include "Thread_Pool.hpp"
#include "Timer.hpp"

#define NUM_THREADS 4

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
        std::vector<int8_t> tensor(tensor_size);
        std::vector<int8_t> mat_1(mat_size);
        std::vector<int8_t> mat_2(mat_size);
        std::vector<int8_t> mat_3(mat_size);

        fill_vec(tensor);

        argmax_tensor(tensor.data(), mat_1.data(), num_filters, mat_size);
        argmax_tensor_mt_thread_pool(tensor.data(), mat_2.data(), num_filters, mat_size, thread_pool);
        argmax_tensor_mt_async(tensor.data(), mat_3.data(), num_filters, mat_size, num_theads);

        comp_vec(mat_1, mat_2);
        comp_vec(mat_1, mat_3);

        std::cout<<"I : "<< i<<" | ";
        // std::cout<<"T : "<< num_theads<<" | ";
        // std::cout<<"C : "<< num_columns<<" | ";
        // std::cout<<"R : "<< num_rows<<" | ";
        // std::cout<<"F : "<< num_filters<<std::endl;
    }
}

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


int main()
{
    test_argmax_mt();
    argmax_mt_benchmark_tp(224, 224, 21, 1000);
    argmax_mt_benchmark_as(224, 224, 21, 1000);
    argmax_mt_benchmark(224, 224, 21, 1000);
    argmax_st_benchmark(224, 224, 21, 1000);

    Timer::Get().print_duration();

    return 0;
}