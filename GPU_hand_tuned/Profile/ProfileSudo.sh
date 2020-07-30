#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

FileName=$2
LogFileName=$FileName'.log'
ProfFileName=$FileName'.nvprof'

rm -f $LogFileName

echo "--- SUMMARY ---" >> $LogFileName
echo >> $LogFileName

sudo /usr/local/cuda-10.0/bin/nvprof --profile-api-trace none ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName


#echo "--- GENERATE FILE FOR VISUAL PROFILER ---" >> $LogFileName
#echo >> $LogFileName

#sudo /usr/local/cuda-10.0/bin/nvprof --kernels :::1 --analysis-metrics -o $ProfFileName -f ./$1 2>>$LogFileName

#echo >> $LogFileName
#echo >> $LogFileName


echo "--- SPECIFIC METRICS AND EVENTS ---" >> $LogFileName
echo >> $LogFileName

sudo /usr/local/cuda-10.0/bin/nvprof --kernels :::1 --events elapsed_cycles_sm,active_cycles --metrics sm_efficiency,achieved_occupancy,eligible_warps_per_cycle,branch_efficiency,tex_cache_throughput,dram_read_throughput,dram_write_throughput,gst_throughput,gld_throughput,local_load_throughput,local_store_throughput,gld_transactions_per_request,gst_transactions_per_request,shared_load_throughput,l1_cache_global_hit_rate,l1_cache_local_hit_rate,shared_store_throughput,l2_read_throughput,l2_write_throughput,l2_l1_read_throughput,l2_l1_write_throughput,l2_texture_read_throughput,gld_efficiency,gst_efficiency,shared_efficiency,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,inst_integer,inst_compute_ld_st,flop_dp_efficiency,l1_shared_utilization,l2_utilization,tex_utilization,dram_utilization,sysmem_utilization,ldst_fu_utilization,alu_fu_utilization,tex_fu_utilization,stall_pipe_busy,stall_exec_dependency,stall_memory_dependency,stall_inst_fetch,stall_texture,stall_not_selected,stall_constant_memory_dependency,stall_memory_throttle,stall_sync,stall_other ./$1 2>>$LogFileName

echo >> $LogFileName
echo >> $LogFileName