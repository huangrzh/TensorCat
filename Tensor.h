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
		void randu(const arma::uvec& sizei);
		void randu(const std::vector<arma::uword>& sizei);
		void randn(const std::vector<arma::uword>& sizei);


	// zeros
		void zeros();



	// MPO--Heisenberg model
		void edge_i_mpo();
		void edge_f_mpo();
		void mpo();



	//------------------------------------------
	//   IO
	//-----------------------------------------
		void print(std::string);
		void print();



	//-------------------------------------------
	// Tensor operations
	//-------------------------------------------
		void reset(const std::vector<arma::uword>& sizei);
		void reset(const arma::mat& datai, const arma::uvec& sizei);
		void reset(const arma::mat& datai, const std::vector<arma::uword>& sizei);
		void reset(const arma::mat& datai);


		void sum_ind(arma::uword i_ind);


		Tensor& operator=(const Tensor&);
		void reshape(const arma::uvec& new_size);
		void reshape(const std::vector<arma::uword>& new_size);
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

	
		Tensor& double_tensor(const Tensor& T1, const Tensor& T2);

		friend double dot(const Tensor& T1, const Tensor& T2);
		


	//-----------------------------------------
	// Tensor element access
	//-----------------------------------------
		double& operator()(const std::vector<arma::uword>& inds);
		double& operator()(arma::uword iele);

		double& at(const arma::uvec& inds);
		double& at(const std::vector<arma::uword>& inds);
		double& at(const arma::uword *inds);
		double& at(arma::uword iele);

		const double& c_at(const arma::uvec& inds)const;
		const double& c_at(const std::vector<arma::uword>& inds)const;
		const double& c_at(const arma::uword *inds)const;
		const double& c_at(arma::uword iele)const;

		const arma::mat& c_data()const;
		arma::mat& data();

		const arma::uvec& size()const;
		arma::uword size(arma::uword i_ind);
		
		


	//------------------------------------------------
	// Attributes:
	//-----------------------------------------------
		const arma::uword ndims;
		const arma::uword n_elem;

		arma::uvec accu_size_;
	private:
		// Attributes:
		arma::mat ele_;
		arma::uvec size_;



		void set_accusize();
		
	};


}


#endif

