#ifndef TENSOR_H
#define TENSOR_H

#include <armadillo>
#include <vector>
#include <string>
#include "setting.h"

namespace TNSCat{

	class Tensor{
	public:
		Tensor();
		Tensor(const Tensor&);
		Tensor(arma::uvec size, arma::uword rank);
		~Tensor();	
		



	// Random tensor
		void randomize(const arma::uvec& sizei);
		void randomize(const std::vector<arma::uword>& sizei);




	//------------------------------------------
	//   IO
	//-----------------------------------------
		void print(std::string);
		void print();



	//-------------------------------------------
	// Tensor operations
	//------------------------------------------
		void reset(const arma::mat& datai, const arma::uvec& sizei);
		void reset(const arma::mat& datai, const std::vector<arma::uword>& sizei);
		void reset(const arma::mat& datai);



		Tensor& operator=(const Tensor&);
		void reshape(const arma::uvec& new_size);
		void tensor_reshape(const std::vector<arma::uvec>& new_inds, const arma::uword* full_oldind, const arma::uword* full_newind);



		void permute(const std::vector<arma::uword>& permutation);
		void permute(const arma::uword* permutation);
		void tensor_permute(const arma::uword* oldorder, const arma::uword* neworder);


		
		Tensor& tensor_product(const std::vector<arma::uword>& final_order, const arma::uword& num_con_inds, const Tensor& T1_, 
			const std::vector<arma::uword>& order1, const std::vector<arma::uword>& forder1, const Tensor& T2_, 
			const std::vector<arma::uword>& order2, const std::vector<arma::uword>& forder2);


		Tensor& tensor_product(const arma::uword& num_con_inds, const Tensor& T1_,
			const std::vector<arma::uword>& order1, const std::vector<arma::uword>& forder1, const Tensor& T2_,
			const std::vector<arma::uword>& order2, const std::vector<arma::uword>& forder2);

		
		Tensor& tensor_product(const arma::uword* final_order, const arma::uword& num_con_inds, const Tensor& T1_, 
			const arma::uword* order1, const arma::uword* forder1, const Tensor& T2_, const arma::uword* order2, const arma::uword* forder2);
		

		Tensor& tensor_product(const arma::uword& num_con_inds, const Tensor& T1_, const arma::uword* order1, 
			const arma::uword* forder1, const Tensor& T2_, const arma::uword* order2, const arma::uword* forder2);

	

		friend double dot(const Tensor& T1, const Tensor& T2);


	//-----------------------------------------
	// Tensor element access
	//-----------------------------------------
		double& at(const arma::uvec& inds);
		double& at(const std::vector<arma::uword>& inds);
		double& at(const arma::uword *inds);
		double& at(arma::uword iele);
		const arma::mat& c_data();
		arma::mat& data();
		arma::mat copy_data();
		arma::uvec& size();
		
		


	//------------------------------------------------
	// Attributes:
	//-----------------------------------------------
		const arma::uword ndims;
		const arma::uword num_ele;

		arma::uvec accu_size_;
	private:
		// Attributes:
		arma::mat ele_;
		arma::uvec size_;



		void set_accusize();
		
	};


}


#endif

