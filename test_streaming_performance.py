#!/usr/bin/env python
# Test to identify where streaming is spending time

import time
import sys
import bioread

def time_operation(name, func):
    """Time an operation and print results"""
    start = time.time()
    result = func()
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s")
    return result, elapsed

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_streaming_performance.py <file.acq>")
        sys.exit(1)

    filename = sys.argv[1]

    print(f"\nTesting file: {filename}\n")
    print("=" * 60)

    # Test 1: Full file read
    print("\n1. FULL FILE READ (stream=False)")
    print("-" * 60)
    data_full, t_full = time_operation(
        "Total time",
        lambda: bioread.read(filename, stream=False)
    )
    print(f"Channels: {len(data_full.channels)}")
    if data_full.channels:
        print(f"Sample rate: {data_full.samples_per_second:.1f} Hz")
        print(f"Total samples: {data_full.channels[0].point_count}")
        print(f"Duration: {data_full.channels[0].point_count / data_full.samples_per_second:.1f}s")

    # Test 2: Streaming mode (default 10 seconds)
    print("\n2. STREAMING MODE (stream=True, default 10s)")
    print("-" * 60)
    data_stream, t_stream = time_operation(
        "Total time",
        lambda: bioread.read(filename, stream=True)
    )
    print(f"Channels: {len(data_stream.channels)}")
    if data_stream.channels:
        print(f"Samples read: {data_stream.channels[0].point_count}")
        print(f"Duration: {data_stream.channels[0].point_count / data_stream.samples_per_second:.1f}s")

    # Test 3: Streaming mode with explicit sample_count
    print("\n3. STREAMING MODE (stream=True, explicit 10s)")
    print("-" * 60)
    sample_count_10s = int(data_full.samples_per_second * 10)
    data_stream_explicit, t_stream_explicit = time_operation(
        "Total time",
        lambda: bioread.read(filename, stream=True, sample_count=sample_count_10s)
    )
    print(f"Samples read: {data_stream_explicit.channels[0].point_count}")

    # Test 4: Just reading headers
    print("\n4. READING HEADERS ONLY")
    print("-" * 60)
    headers, t_headers = time_operation(
        "Header read time",
        lambda: bioread.read_headers(filename)
    )

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"Full file read:           {t_full:.3f}s")
    print(f"Stream (default 10s):     {t_stream:.3f}s  ({t_stream/t_full*100:.1f}% of full read)")
    print(f"Stream (explicit 10s):    {t_stream_explicit:.3f}s  ({t_stream_explicit/t_full*100:.1f}% of full read)")
    print(f"Header read only:         {t_headers:.3f}s  ({t_headers/t_full*100:.1f}% of full read)")
    print(f"Data read time (stream):  {t_stream - t_headers:.3f}s")

    if t_stream > t_full:
        print(f"\n⚠️  WARNING: Streaming is SLOWER than full read by {t_stream - t_full:.3f}s!")
        print("This suggests an inefficiency in the streaming code path.")
    else:
        print(f"\n✓ Streaming is faster by {t_full - t_stream:.3f}s")

    # Calculate expected ratio
    if data_full.channels:
        full_duration = data_full.channels[0].point_count / data_full.samples_per_second
        stream_duration = data_stream.channels[0].point_count / data_stream.samples_per_second
        expected_ratio = stream_duration / full_duration
        actual_ratio = t_stream / t_full

        print(f"\nExpected time ratio (data amount): {expected_ratio:.2f}x")
        print(f"Actual time ratio:                  {actual_ratio:.2f}x")
        if actual_ratio > expected_ratio * 1.5:
            print("⚠️  Streaming has significant overhead beyond just reading less data!")
