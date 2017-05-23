#include <iostream>
#include <iomanip>
#include <cmath>
#include "test.h"


void TNSCat::test_rand_tensor(){
	std::cout << __FUNCTION__ << std::endl;
	Tensor t1;
	arma::uvec size1(4);
	size1 << 2 << 3 << 4 << 5;
	double time0 = TNS_TIME;
	t1.randomize(size1);
	double time1 = TNS_TIME;
	std::cout << "Time " << std::setprecision(5) << time1 - time0 << "s" << std::endl;

	t1.print();


	arma::uword inds[4] = { 1, 2, 3, 4 };
	std::cout << "Ele = " << t1.at(inds) << std::endl;
	t1.at(inds) = 1.0;
	std::cout << "Ele = " << t1.at(inds) << std::endl;
}




void TNSCat::test_reshape(){
	std::cout << __FUNCTION__ << std::endl;
	
	Tensor t1;
	arma::uvec size1(6);
	size1 << 2 << 3 << 4 << 5 << 8 << 9;
	
	t1.randomize(size1);
	t1.print();

	arma::uvec newsize(2);
	newsize << 24 << 360;

	double time0 = TNS_TIME;
	Tensor t2(t1);
	t2.reshape(newsize); 
	double time1 = TNS_TIME;
	std::cout << "Tensor reshape time " << std::setprecision(5) << time1 - time0 << "s" << std::endl;
	
	t2.print();	
	

	double ele_err = 0.;
	arma::uword iele = 0;
	for (arma::uword i6 = 0; i6 < 9; i6++){
		for (arma::uword i5 = 0; i5 < 8; i5++){
			for (arma::uword i4 = 0; i4 < 5; i4++){
				for (arma::uword i3 = 0; i3 < 4; i3++){
					for (arma::uword i2 = 0; i2 < 3; i2++){
						for (arma::uword i1 = 0; i1 < 2; i1++){
							arma::uword inds[6] = { i1, i2, i3, i4, i5, i6 };
							ele_err += std::fabs(t1.at(inds) - t2.at(iele));
							iele++;
						}
					}
				}
			}
		}
	}
	std::cout << "ele_err = " << std::setprecision(6) << ele_err << std::endl;
}







void TNSCat::test_premute(){
	std::cout << __FUNCTION__ << std::endl;
	Tensor t1;
	arma::uvec size1(6);
	size1 << 2 << 3 << 4 << 5 << 8 << 9;

	t1.randomize(size1);
	//t1.print("Tensor t1");


	Tensor t2(t1);
	arma::uword per[6] = { 1, 3, 0, 2, 5, 4 };
	t2.permute(per);
	
	double ele_err = 0.;
	arma::uword iele = 0;
	for (arma::uword i6 = 0; i6 < 9; i6++){
		for (arma::uword i5 = 0; i5 < 8; i5++){
			for (arma::uword i4 = 0; i4 < 5; i4++){
				for (arma::uword i3 = 0; i3 < 4; i3++){
					for (arma::uword i2 = 0; i2 < 3; i2++){
						for (arma::uword i1 = 0; i1 < 2; i1++){
							arma::uword inds1[6] = { i1, i2, i3, i4, i5, i6 };
							arma::uword inds2[6] = { i2, i4, i1, i3, i6, i5 };							
							ele_err += std::fabs(t1.at(inds1) - t2.at(inds2));
							iele++;
						}
					}
				}
			}
		}
	}
	std::cout << "Ele Err1 = " << std::setprecision(6) << ele_err << std::endl;




	Tensor t3(t1);
	arma::uword order1[6] = { 0, 1, 2, 3, 4, 5 };
	arma::uword order2[6] = { 1, 3, 0, 2, 5, 4 };
	t3.tensor_permute(order1, order2);

	ele_err = 0.;
	iele = 0;
	for (arma::uword i6 = 0; i6 < 9; i6++){
		for (arma::uword i5 = 0; i5 < 8; i5++){
			for (arma::uword i4 = 0; i4 < 5; i4++){
				for (arma::uword i3 = 0; i3 < 4; i3++){
					for (arma::uword i2 = 0; i2 < 3; i2++){
						for (arma::uword i1 = 0; i1 < 2; i1++){
							arma::uword inds1[6] = { i1, i2, i3, i4, i5, i6 };
							arma::uword inds3[6] = { i2, i4, i1, i3, i6, i5 };
							ele_err += std::fabs(t1.at(inds1) - t3.at(inds3));
							iele++;
						}
					}
				}
			}
		}
	}
	std::cout << "Ele Err2 = " << std::setprecision(6) << ele_err << std::endl;
}






void TNSCat::test_tensor_product(){
	std::cout << __FUNCTION__ << std::endl;
	Tensor t1;
	arma::uvec size1(6);
	size1 << 2 << 3 << 4 << 5 << 8 << 9;

	t1.randomize(size1);
	//t1.print("Tensor t1");


	Tensor t2(t1);
	arma::uword per[6] = { 1, 3, 0, 2, 5, 4 };
	t2.permute(per);
	//t2.print("Tensor t2");

	double time1 = TNS_TIME;
	Tensor t3;
	arma::uword final_order[] = { 2, 7, 10, 3, 4, 5, 9, 8 };
	arma::uword order1[] = { 1, 2, 3, 4, 5, 6 };
	arma::uword forder1[] = { 2, 3, 4, 5, 1, 6 };
	arma::uword order2[] = { 7, 8, 1, 9, 6, 10 };
	arma::uword forder2[] = { 1, 6, 7, 9, 8, 10 };
	t3.tensor_product(final_order, 2, t1, order1, forder1, t2, order2, forder2);	
	double time2 = TNS_TIME;
	std::cout << "Tensor product time = " << std::setprecision(3) << time2 - time1 << "s" << std::endl;
	//t3.print("Tensor t3");
	//system("pause");

	t1.tensor_permute(order1, forder1);
	t2.tensor_permute(order2, forder2);
	arma::mat mat1 = t1.data();
	arma::mat mat2 = t2.data();
	mat1.reshape(t1.num_ele / 18, 18);
	mat2.reshape(18, t2.num_ele / 18);
	arma::mat mat3 = mat1*mat2;


	arma::uword inds4[] = { 3, 4, 5, 8, 3, 4, 5, 8 };
	arma::uvec inds4_(inds4, 8);
	Tensor t4(inds4_, 8);
	t4.data() = mat3;
	arma::uword old_order[] = { 2, 3, 4, 5, 7, 9, 8, 10 };
	t4.tensor_permute(old_order, final_order);

	arma::uword numele3 = mat3.n_elem;
	double err3 = 0.;
	for (arma::uword iele = 0; iele < numele3; iele++){
		err3 += std::fabs(t4.at(iele) - t3.at(iele));
	}
	std::cout << "Err3 = " << std::setprecision(6) << err3 << std::endl;
}