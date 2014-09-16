void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + block_x - 1)/block_x;
        grid.y = (NY + block_y - 1)/block_y;
}

void setCoarsePermIntegrationArgs(CoarsePermIntegrationKernelArgs* args, GpuPtrRaw K, GpuPtrRaw k_data, GpuPtrRaw k_heights, GpuPtrRaw p_cap_ref_table, GpuPtrRaw s_b_ref_table, unsigned int nx, unsigned int ny, unsigned int border){
	args->K = K;
	args->k_data = k_data;
	args->k_heights = k_heights;
	args->p_cap_ref_table = p_cap_ref_table;
	args->s_b_ref_table = s_b_ref_table;

	args->nx = nx;
	args->ny = ny;
	args->border = border;
}
