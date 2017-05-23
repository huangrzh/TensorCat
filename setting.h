#ifndef SETTING_H
#define SETTING_H


#define VISUAL
//#define LINUX

#define DEBUG_CODE


//#define MPITNS
#ifdef MPITNS
	#define TNS_TIME (MPI_Wtime())
#else
	#define TNS_TIME (clock()*1.0/CLOCKS_PER_SEC)
#endif

#endif
