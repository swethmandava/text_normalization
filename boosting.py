import xgboost as xgb

#Skeleton code for xgboost.
#Include test parsing and perform cross validation

def train():
	dtrain = xgb.DMatrix(train_x, label=train_y)
	#Can include validation for early stopping
	dtest = xgb.DMatrix(test_x)

	params = {}
	params['eta'] = 0.1 #learning rate
	params['gamma'] = 0.01 #Loss to split further
	params['max_depth'] = 2 #Maximum depth of tree
	params['subsample'] = 0.5 #percentage of batch to use in each epoch
	params['lambda'] = 0.01 #L2 regularization
	params['alpha'] = 0 #L1 regularization
	params['scale_pos_weight'] = 0.1
	# We should look into this for sure. It's used for unbalanced weights
	# Typical value is sum(negative cases)/sum(positive cases)

	#There are a ton more parameters we could experiment with here
	# xgboost.readthedocs.io/en/latest/parameter.html

	evallist = [(dtest, 'eval'), (dtrain, 'train')]
	bst = xgb.train(param, dtrain, epochs, evallist)
	bst.save_model('best_model.model')
	bst.plot_importance(bst) #plots histogram showing importance of features
	bst.plot_tree(bst, num_trees=2) #plots 2 trees

if __name__ == '__main__':
	train()