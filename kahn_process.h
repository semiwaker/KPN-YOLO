#ifndef	KAHN_PROCESS_H
#define	KAHN_PROCESS_H
/*
 *	kahn_process.h -- Base SystemC module for modeling applications using KPN
 *
 */

#include <systemc.h>

class	kahn_process : public sc_module
{
	public:

	SC_HAS_PROCESS(kahn_process);

	kahn_process(sc_module_name name) : sc_module(name)
	{
		iter = 0;
		SC_THREAD(main);
	}

	void	main()	{ while(true) {process(); iter++;} }

	protected:

	int iter = 0;

	virtual void process() = 0;
};
#endif
