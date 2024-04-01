import os
import sys
from student_impl.eid_ch48458 import A_Star_Search
eid = "ch48458"
idx = int(sys.argv[1])
benchmark_path = f"benchmarks/example_{idx}.txt"
output_root = "output"
output_root = os.path.join(output_root, eid)
if not os.path.isdir(output_root):
   os.mkdir(output_root)

output_path = os.path.join(output_root, os.path.basename(benchmark_path))
solver = A_Star_Search()
solver.read_benchmark(benchmark_path)
solution = solver.solve()
solver.plot_solution(solution, os.path.join(output_root, f"example_{idx}_sol.png"))
profiling = solver.profile(n_runs=3) # ignore memory for now. runtime will be graded
solver.dump_output_file(*solution, *profiling, output_path)
