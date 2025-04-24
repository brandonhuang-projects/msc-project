"""
This script is written to test the interface of SharedMemory library written for Julia

Author: Benyamin Izadpanah
Copyright: Benyamin Izadpanah
Github Repository: https://github.com/ben-izd/shared_memory
Start Date: 2022-8
Last date modified: 2022-11
Version used for testing: Julia 1.8.3

Requirement:
    - Make sure run this file in ".\\test-framework\\interface" directory (depends on @__DIR__)
"""

using Random
using Test

include(joinpath(@__DIR__,"..","..","julia","shared_memory.jl"))
using .shared_memory

file_path = joinpath(@__DIR__,"julia_interface_test_data")


JULIA_TYPES= [UInt8, UInt16, UInt32,UInt64,Int8, Int16, Int32,Int64,Float32,Float64,ComplexF32,ComplexF64]

@testset "Utilities 1" begin
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_data_type()
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_rank()
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_flatten_length()
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_dimensions()
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_flatten_data()
    @test_throws SharedMemoryLibraryError(-1) get_shared_memory_data()
end

set_shared_memory_path(file_path)

# Test Sharing Data
@testset "fixed-size" begin

    function generate_case(type::DataType,size::Tuple)
        data = rand(type, size)
        set_shared_memory_data(data)

        @test get_shared_memory_data_type()     == type
        @test get_shared_memory_rank()          == length(size)
        @test get_shared_memory_dimensions()    == collect(size)
        @test get_shared_memory_flatten_length()== prod(size)
        @test get_shared_memory_data()          == data
    end

    for type = JULIA_TYPES
        generate_case(type, (5,))
        generate_case(type, (6,4))
        generate_case(type, (3,5,7))
    end

    # Testing String
    sample_text = "Julia 😍 julia 😉 Julia 😁";
    set_shared_memory_data(sample_text)
    @test get_shared_memory_data_type()     == String
    @test get_shared_memory_rank()          == 1
    @test get_shared_memory_dimensions()    == [32]
    @test get_shared_memory_flatten_length()== 32
    @test get_shared_memory_data()          == sample_text
end


@testset "Random-Size" begin
    function generate_case(type::DataType,rank)
        
        # size = convert(Vector{Int64},floor.(rand(Uniform(5,10),rank)))
        size = Int64.(trunc.(rand(rank) * 5)) .+ 5

        data = rand(type, Tuple(size))
        set_shared_memory_data(data)

        @test get_shared_memory_data_type()     == type
        @test get_shared_memory_rank()          == rank
        @test get_shared_memory_dimensions()    == size
        @test get_shared_memory_flatten_length() == prod(size)
        @test get_shared_memory_data()          == data

    end

    for type = JULIA_TYPES
        generate_case(type, 2)
        generate_case(type, 5)
        generate_case(type, 7)
    end
end

# Rest the state
delete_shared_memory()

# Utilities
@testset "Utilities 2" begin
    @test !isfile(file_path)

    set_shared_memory_data([1])

    @test isfile(file_path)

    @assert isfile(file_path)
    rm(file_path)

    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_data_type()
    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_rank()
    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_flatten_length()
    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_dimensions()
    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_flatten_data()
    @test_throws SharedMemoryLibraryError(-2) get_shared_memory_data()
end

