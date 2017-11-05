#coding:utf-8
#author:baixuefeng
#No. 17S003001


import numpy as np

nn_input_dim = 3
nn_output_dim= 4
nn_hidden = 4
learn_rate=0.0001  # 学习率设置
reg_lambda=0.01  # 正则项

mu = np.array([[0,0,0],[0,1,0],[-1,0,1],[0,0.5,1]])
sigma0 = np.array([[1,0,0],[0,1,0],	[0,0,1]	])
sigma1 = np.array([	[1,0,1],[0,2,2],[1,2,5]])
sigma2 = np.array([[2,0,0],[0,6,0],[0,0,1]])
sigma3 = np.array([[2,0,0],[0,1,0],[0,0,3]])

def generate_data(sampleNo):
	'''生成数据'''
	x1 = np.random.multivariate_normal(mu[0],sigma0,sampleNo)
	x2 = np.random.multivariate_normal(mu[1],sigma1,sampleNo)
	x3 = np.random.multivariate_normal(mu[2],sigma2,sampleNo)
	x4 = np.random.multivariate_normal(mu[3],sigma3,sampleNo)
	X = np.concatenate((x1,x2,x3,x4))
	y = np.array([index//sampleNo for index in range(sampleNo*4)])
	return X,y

def shuffle(X,y):
	'''shuffle 数据'''
	shuff_index = range(X.shape[0])
	np.random.shuffle(shuff_index)
	return X[shuff_index],y[shuff_index]

def calcu_loss(model):

	'''计算Loss:交叉熵+正则项'''
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	# 正向传播
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	# loss计算
	corect_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(corect_logprobs)

	# 对W的正则项约束
	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1./num_examples * data_loss

def predict(model, x):
	'''预测'''
	
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	# 正向计算
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	
	return np.argmax(probs, axis=1),probs


def train_model(Threshold=1e-5):
	'''训练'''
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hidden) / np.sqrt(3) #w~N(0,0.25)
	b1 = np.zeros((1, nn_hidden))
	W2 = np.random.randn(nn_hidden, nn_output_dim) / np.sqrt(4)
	b2 = np.zeros((1, nn_output_dim))
	model = {}
	i = 0
	previous = 0.0
	newloss = 10.0
	while np.fabs(newloss-previous)>= Threshold:
		i+=1
		previous = newloss
		# 正向传播
		z1 = X.dot(W1)+b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2)+b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# 反向传播
		delta3 = probs
		delta3[range(num_examples), y] -= 1  #对net的偏导
		dW2 = (a1.T).dot(delta3)			 #对W2的偏导
		db2 = np.sum(delta3,axis=0, keepdims=True) #对b2的偏导
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)

		# 正则项
		dW2 += reg_lambda * W2
		dW1 += reg_lambda * W1

		# 梯度下降参数更新

		W1 += -learn_rate * dW1
		b1 += -learn_rate * db1
		W2 += -learn_rate * dW2
		b2 += -learn_rate * db2

		# 保存模型
		model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
		newloss = calcu_loss(model)
		# 打印Loss
		if i % 1000 == 0:
			print "Loss after iteration #%i: %f" %(i, calcu_loss(model))


	return model


if __name__ =="__main__":
	
	trainX,trainY = generate_data(1000)
	testX,testY = generate_data(100)
	X,y = shuffle(trainX,trainY)
	tX,ty = shuffle(testX,testY)
	num_examples = len(X)

	print("TraningSet shape X:{},Y:{}".format(X.shape,y.shape))
	print("Training... Threshold={}".format(1e-5))

	model = train_model()
	train_res,probs = predict(model,trainX)
	accuracy0 = 1.0*np.sum([1 if train_res[i]==trainY[i] else 0 for i in range(len(train_res))])/len(trainY)
	print("train Accuracy: {}%".format(accuracy0*100))

	test_res,probs = predict(model,tX)
	accuracy = 1.0*np.sum([1 if test_res[i]==ty[i] else 0 for i in range(len(test_res))])/len(ty)
	print("test Accuracy: {}%".format(accuracy*100))
