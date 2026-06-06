from rag_eval.benchmarks.benchmark_generator import Benchmark, LexicalOverlapAnalyzer

bm = Benchmark.load("your_existing_benchmark.json")
analyzer = LexicalOverlapAnalyzer()
stats = analyzer.analyze_benchmark(bm)
print(stats)
