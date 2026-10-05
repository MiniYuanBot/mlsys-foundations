// Original teaching scaffold, no third-party source copied.
#include "tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

static std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::out_of_range("offset multiplication overflow");
    return a * b;
}
static std::size_t checked_sum(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::out_of_range("offset addition overflow");
    return a + b;
}
static float ref_element(const View& v, std::size_t i, std::size_t j) {
    if (i >= v.rows || j >= v.cols) throw std::out_of_range("logical bounds");
    const auto index = checked_sum(checked_product(i, v.row_stride),
                                   checked_product(j, v.col_stride));
    if (index >= v.storage_size) throw std::out_of_range("storage bounds");
    return v.data[index];
}
static std::vector<float> reference(const View& a, const View& b) {
    if (!a.rows || !a.cols || !b.cols || a.cols != b.rows) throw std::invalid_argument("shape");
    std::vector<float> c(checked_product(a.rows, b.cols));
    for (std::size_t i=0; i<a.rows; ++i)
        for (std::size_t j=0; j<b.cols; ++j) {
            double total=0;
            for (std::size_t k=0; k<a.cols; ++k)
                total += static_cast<double>(ref_element(a,i,k))*ref_element(b,k,j);
            c[i*b.cols+j]=static_cast<float>(total);
        }
    return c;
}
static void close(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size()!=expected.size()) throw std::runtime_error("shape mismatch");
    for (std::size_t i=0; i<actual.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i]) > 1e-4f+1e-4f*std::abs(expected[i]))
            throw std::runtime_error("numeric mismatch at "+std::to_string(i));
}
static void tests(bool supplied_only) {
    std::vector<float> storage={1,2,3,99,4,5,6,99};
    View padded{storage.data(),2,3,4,1,storage.size()};
    View transposed{storage.data(),3,2,1,4,storage.size()};
    auto read=supplied_only?ref_element:element;
    if(read(padded,1,2)!=6 || read(transposed,2,1)!=6) throw std::runtime_error("strides");
    bool rejected=false;
    try { (void)read(padded,2,0); } catch(const std::out_of_range&) { rejected=true; }
    if(!rejected) throw std::runtime_error("bounds not rejected");
    auto rejects_address = [&](const View& view, std::size_t row, std::size_t col,
                               const char* reason) {
        bool address_rejected=false;
        try { (void)read(view,row,col); }
        catch(const std::out_of_range&) { address_rejected=true; }
        if(!address_rejected) throw std::runtime_error(reason);
    };
    // All logical coordinates are valid. The address calculation must reject
    // a too-short declared storage and both multiply/add overflow before access.
    View truncated=padded;
    truncated.storage_size=6;
    rejects_address(truncated,1,2,"storage bounds not rejected");
    const auto maximum=std::numeric_limits<std::size_t>::max();
    View multiply_overflow{storage.data(),3,1,maximum/2+1,1,storage.size()};
    View addition_overflow{storage.data(),2,2,maximum,1,storage.size()};
    rejects_address(multiply_overflow,2,0,"offset multiplication overflow not rejected");
    rejects_address(addition_overflow,1,1,"offset addition overflow not rejected");
    auto fn=supplied_only?reference:matmul;
    close(fn(padded,transposed), {14,32,32,77});
    const auto before=storage;
    close(fn(transposed,padded), reference(transposed,padded));
    if(storage!=before) throw std::runtime_error("input changed");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1,1);
    for(auto m:{1,7,31}) for(auto n:{1,9,17}) {
        const std::size_t k=13;
        std::vector<float> a(m*(k+2)), b(k*(n+3));
        for(auto& x:a) x=uniform(rng);
        for(auto& x:b) x=uniform(rng);
        View av{a.data(),static_cast<std::size_t>(m),k,k+2,1,a.size()};
        View bv{b.data(),k,static_cast<std::size_t>(n),static_cast<std::size_t>(n+3),1,b.size()};
        close(fn(av,bv),reference(av,bv));
    }
    rejected=false;
    try { (void)fn(padded,padded); } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid matmul shape accepted");
    rejected=false;
    try { (void)fn(truncated,transposed); } catch(const std::out_of_range&) { rejected=true; }
    if(!rejected) throw std::runtime_error("matmul accepted invalid storage extent");
}
static void benchmark() {
    tests(false); // No timing an unimplemented/incorrect operator.
    std::cout<<"implementation,M,N,K,repeat,iterations,latency_ms,checksum\n";
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1,1);
    volatile float sink=0;
    for(auto size:{64,128,256}) {
        std::vector<float> a(size*size), b(size*size);
        for(auto& x:a) x=uniform(rng);
        for(auto& x:b) x=uniform(rng);
        View av{a.data(),static_cast<std::size_t>(size),static_cast<std::size_t>(size),
                static_cast<std::size_t>(size),1,a.size()};
        View bv{b.data(),static_cast<std::size_t>(size),static_cast<std::size_t>(size),
                static_cast<std::size_t>(size),1,b.size()};
        const auto original_a=a, original_b=b;
        const auto expected=reference(av,bv);
        auto check_inputs = [&]() {
            if(a!=original_a || b!=original_b)
                throw std::runtime_error("benchmark input changed");
        };
        for(auto label:{"reference","student"}) {
            auto fn=std::string(label)=="reference"?reference:matmul;
            // Validate this exact timed shape, not just the small contract cases.
            close(fn(av,bv),expected);
            check_inputs();
            for(int i=0;i<3;++i) sink=fn(av,bv)[0];
            check_inputs();
            for(int repeat=1;repeat<=3;++repeat) {
                std::vector<float> output;
                const auto begin=std::chrono::steady_clock::now();
                for(int i=0;i<10;++i) {
                    output=fn(av,bv);
                    sink=output[0];
                }
                const auto end=std::chrono::steady_clock::now();
                const double ms=std::chrono::duration<double,std::milli>(end-begin).count()/10;
                // Verification and full-output checksum are outside timing.
                close(output,expected);
                check_inputs();
                const double checksum=std::accumulate(output.begin(),output.end(),0.0);
                std::cout<<label<<","<<size<<","<<size<<","<<size<<","<<repeat<<",10,"<<ms<<","<<checksum<<"\n";
            }
        }
    }
    (void)sink;
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("use --test, --reference-test, or --benchmark");
        const std::string mode=argv[1];
        if(mode=="--test") { tests(false); std::cout<<"student contract passed\n"; }
        else if(mode=="--reference-test") { tests(true); std::cout<<"supplied reference passed (not student credit)\n"; }
        else if(mode=="--benchmark") benchmark();
        else throw std::invalid_argument("unknown mode");
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; return 1; }
}
