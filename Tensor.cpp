#include <iomanip>
#include "Tensor.h"


TNSCat::Tensor::Tensor() 
	: num_ele(0),
	  ndims(0){
}




TNSCat::Tensor::Tensor(const Tensor& T)	: num_ele(T.num_ele), ndims(T.ndims){
	ele_ = T.ele_;
	size_ = T.size_;
	accu_size_ = T.accu_size_;
}



TNSCat::Tensor::Tensor(arma::uvec sizei, arma::uword rank) : num_ele(arma::prod(sizei)), ndims(rank){
	size_ = sizei;
	ele_.set_size(num_ele);
	set_accusize();
}



TNSCat::Tensor::~Tensor(){}







void TNSCat::Tensor::print(){
	std::cout << std::endl << "------------------------" << std::endl;
	std::cout << "Rank : " << ndims << std::endl;
	std::cout << "NumOfEle : " << num_ele << std::endl;
	std::cout << "Size : " << std::endl << size_ << std::endl;
	std::cout << "AccuSize : " << std::endl << accu_size_ << std::endl;
	std::cout << "Data Size : " << ele_.n_rows << ", " << ele_.n_cols << std::endl;
	std::cout << "------------------------" << std::endl << std::endl;
}



void TNSCat::Tensor::print(std::string s){
	std::cout << s << std::endl;
	print();
}



void TNSCat::Tensor::randomize(const arma::uvec& sizei){
	const_cast<arma::uword&>(num_ele) = arma::prod(sizei);
	const_cast<arma::uword&>(ndims) = sizei.n_elem;
	ele_ = arma::randu(num_ele);
	size_ = sizei;
	accu_size_ = size_;
	accu_size_(0) = 1;
	accu_size_(1) = size_(0);
	for (arma::uword ind = 2; ind < ndims; ind++){
		accu_size_(ind) = accu_size_(ind - 1)*size_(ind - 1);
	}
}






TNSCat::Tensor& TNSCat::Tensor::operator=(const Tensor& T){
	const_cast<arma::uword&>(num_ele) = T.num_ele;
	const_cast<arma::uword&>(ndims) = T.ndims;
	ele_ = T.ele_;
	size_ = T.size_;
	accu_size_ = T.accu_size_;
	return *this;
}







void TNSCat::Tensor::reshape(const arma::uvec& new_size){
#ifdef DEBUG_CODE
	arma::uword numin = arma::prod(new_size);
	if (numin != num_ele){
		std::cout << "Error in Tensor::reshape: size doesn't match!" << std::endl;
		return;
	}
#endif 
	arma::uword ndimin = new_size.n_elem;
	const_cast<arma::uword&>(ndims) = ndimin;
	size_ = new_size;
	accu_size_ = size_;
	accu_size_(0) = 1;
	accu_size_(1) = size_(0);
	for (arma::uword ind = 2; ind < ndims; ind++){
		accu_size_(ind) = accu_size_(ind - 1)*size_(ind - 1);
	}
}



void TNSCat::Tensor::tensor_reshape(const std::vector<arma::uvec>& new_inds, const arma::uword* full_oldind, const arma::uword* full_newind){
#ifdef DEBUG_CODE
	if (new_inds.size() > ndims){
		std::cout << "Error in Tensor::tensor_reshape: input new_inds has too many legs!" << std::endl;
		return;
	}
#endif
	bool ifpermute = false;
	arma::uvec oldsize = size_;
	for (arma::uword iorder = 0; iorder < ndims; iorder++){
		if (full_oldind[iorder] != full_newind[iorder]){
			tensor_permute(full_oldind, full_newind);
			break;
		}
	}
	arma::uword nlegs = new_inds.size();
	if (ndims > nlegs){
		arma::uvec newsize(nlegs);
		for (arma::uword ileg = 0; ileg < nlegs; ileg++){
			newsize(ileg) = arma::prod(size_(new_inds[ileg]));
		}
		reshape(newsize);
	}
}





void TNSCat::Tensor::permute(const arma::uword* permutation){
	bool ifpermute = false;
	for (arma::uword iorder = 0; iorder < ndims; iorder++){
		if (permutation[iorder] != iorder){
			ifpermute = true;
			break;
		}
	}

	if (ifpermute){
		arma::uvec oldsize = size_;
		for (arma::uword ind = 0; ind < ndims; ind++){
			int iorder = permutation[ind];
			size_(ind) = oldsize(iorder);
		}

		arma::mat oldele = ele_;
		arma::uvec w0(ndims),
			w1(ndims);
		w0(0) = 0;
		w1(0) = 0;

		arma::uvec old_accu = accu_size_;
		accu_size_ = size_;
		accu_size_(0) = 1;
		accu_size_(1) = size_(0);
		for (arma::uword ind = 2; ind < ndims; ind++){
			accu_size_(ind) = accu_size_(ind - 1)*size_(ind - 1);
		}

		arma::uvec iorder(ndims);
		arma::uword res_iele = 0;
		for (arma::uword iele = 0; iele < num_ele; iele++){
			res_iele = iele;
			for (arma::uword idim = ndims - 1; idim > 0; idim--){
				iorder(idim) = res_iele / old_accu(idim);
				res_iele = res_iele % old_accu(idim);
			}
			iorder(0) = res_iele / old_accu(0);

			arma::uword new_iele = 0;
			for (arma::uword idim = 0; idim < ndims; idim++){
				new_iele += iorder(permutation[idim])*accu_size_(idim);
			}
			ele_(new_iele) = oldele(iele);
		}
	}
}



void TNSCat::Tensor::tensor_permute(const arma::uword* oldorderi, const arma::uword* neworderi){
	double time1 = TNS_TIME;
	arma::uvec oldorder(oldorderi, ndims);
	arma::uvec neworder(neworderi, ndims);
	arma::uvec IX1 = arma::sort_index(oldorder);
	arma::uvec IX2 = arma::sort_index(neworder);
	arma::uvec per(ndims);
	per(IX2) = IX1;
	double time2 = TNS_TIME;
	std::cout << "Tensor permute time1 = " << std::setprecision(3) << time2 - time1 << "s" << std::endl;

	permute(per.memptr());
	double time3 = TNS_TIME;
	std::cout << "Tensor permute time2 = " << std::setprecision(3) << time3 - time2 << "s" << std::endl;
}





TNSCat::Tensor& TNSCat::Tensor::tensor_product(const arma::uword* final_order, const arma::uword& num_con_inds, const Tensor& T1_, const arma::uword* order1, const arma::uword* forder1, const Tensor& T2_, const arma::uword* order2, const arma::uword* forder2){
	double time1 = TNS_TIME;
	tensor_product(num_con_inds, T1_, order1, forder1, T2_, order2, forder2);
	double time2 = TNS_TIME;
	std::cout << "Tensor product time1 = " << std::setprecision(3) << time2 - time1 << "s" << std::endl;

	arma::uvec oldinds(ndims);
	arma::uword ndims1 = T1_.ndims,
		ndims2 = T2_.ndims;
	arma::uword ileg = 0;
	for (arma::uword ind = 0; ind < ndims1-num_con_inds; ind++){
		oldinds(ileg) = forder1[ind];
		ileg++;
	}

	for (arma::uword ind = num_con_inds; ind < ndims2; ind++){
		oldinds(ileg) = forder2[ind];
		ileg++;
	}

	double time3 = TNS_TIME;
	std::cout << "Tensor product time2 = " << std::setprecision(3) << time3 - time2 << std::endl;
	tensor_permute(oldinds.memptr(), final_order);
	double time4 = TNS_TIME;
	std::cout << "Tensor product time3 = " << std::setprecision(3) << time4 - time3 << std::endl;

	return *this;
}






TNSCat::Tensor& TNSCat::Tensor::tensor_product(const arma::uword& num_con_inds, const Tensor& T1_, const arma::uword* order1, const arma::uword* forder1, const Tensor& T2_, const arma::uword* order2, const arma::uword* forder2){
	Tensor T1(T1_),
		T2(T2_);
	T1.tensor_permute(order1, forder1);
	T2.tensor_permute(order2, forder2);
	arma::uword dim_com = arma::prod(T1.size_.subvec(T1.ndims - num_con_inds, T1.ndims - 1));
	T1.ele_.reshape(T1.num_ele / dim_com, dim_com);
	T2.ele_.reshape(dim_com, T2.num_ele / dim_com);

	ele_ = T1.ele_ * T2.ele_;
	const_cast<arma::uword&>(num_ele) = ele_.n_elem;
	const_cast<arma::uword&>(ndims) = T1_.ndims + T2_.ndims - 2 * num_con_inds;

	size_.set_size(ndims);
	arma::uvec size1 = T1.size_,
		size2 = T2.size_;
	arma::uword ileg = 0;
	for (arma::uword ind = 0; ind < T1_.ndims - num_con_inds; ind++){
		size_(ileg) = size1(ind);
		ileg++;
	}

	for (arma::uword ind = num_con_inds; ind < size2.n_elem; ind++){
		size_(ileg) = size2(ind);
		ileg++;
	}
	set_accusize();

	return *this;
}







double& TNSCat::Tensor::at(const arma::uvec& inds){
	arma::uword iele = 0;
	for (arma::uword idim = 0; idim < ndims; idim++){
		iele += accu_size_(idim) * inds(idim);
	}
	return ele_.at(iele);
}





double& TNSCat::Tensor::at(const std::vector<arma::uword>& inds){
	arma::uword iele = 0;
	for (arma::uword idim = 0; idim < ndims; idim++){
		iele += accu_size_(idim) * inds[idim];
	}
	return ele_.at(iele);
}



double& TNSCat::Tensor::at(const arma::uword *inds){
	arma::uword iele = 0;
	for (arma::uword idim = 0; idim < ndims; idim++){
		iele += accu_size_(idim) * inds[idim];
	}
	return ele_.at(iele);
}



double& TNSCat::Tensor::at(arma::uword iele){
	return ele_.at(iele);
}


arma::mat& TNSCat::Tensor::data(){
	return ele_;
}







void TNSCat::Tensor::set_accusize(){
	accu_size_.set_size(ndims);
	accu_size_(0) = 1;
	accu_size_(1) = size_(0);
	for (arma::uword ind = 2; ind < ndims; ind++){
		accu_size_(ind) = accu_size_(ind - 1)*size_(ind - 1);
	}
}