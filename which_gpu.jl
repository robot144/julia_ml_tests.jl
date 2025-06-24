#$ Some prototyping on using a specific GPU on a multigpu system
# Note that newer GPUs, such as the Nvidia A100 can be split up into virual GPUs


using CUDA

# size of problem (tune for hardware)
n=1000 #size of the problem

# on GPU
println("Run 10 MM-mutiplications for n=$(n) on GPU")
a=CUDA.randn(n,n);
b=CUDA.randn(n,n);
for i=1:6
   @time b=a*b;
end

#on CPU
println("Run 10 MM-mutiplications for n=$(n) on CPU")
a_cpu=randn(n,n);
b_cpu=randn(n,n);
for i=1:6
   @time b_cpu=a_cpu*b_cpu;
end
