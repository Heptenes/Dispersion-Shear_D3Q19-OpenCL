__kernel void CPU_sphere_collide(int x)
{
	int i = get_global_id(0);
	x = i+1;
}