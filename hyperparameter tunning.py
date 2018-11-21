from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import tensorflow as tf
from keras import backend as K

## NORMAL TRAINING
def data():
	y_train = train["UNITRULE_CODE"]
	X_train = train.drop(unused_cols, axis = 1)

	y_test = test["UNITRULE"]
	X_test = test.drop(unused_cols, axis = 1)

	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler()

	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, y_train, X_test, y_test

# create model

def create_model(X_train, y_train, X_test, y_test):
	def auc(y_true, y_pred):
		auc = tf.metrics.auc(y_true, y_pred)[1]
		K.get_session().run(tf.local_variables_initializer())
		return auc

	# Create model Architecture
	model = Sequential()
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}, input_dim = X_train.shape[1]))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense(1, activation = "sigmoid"))

	# Comliling
	adam = Adam(lr = 0.0001 , decay = 1e-6)
	model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["acc", auc])

	# Training
	result = model.fit(X_train, y_train,
		epochs = 20
		batch_size = {{choice[64,128]}},
		validation_split = 0.2,
		shuffle = True)

	# Save the best result
	val_auc = np.amax(result.history["val_auc"])
	print(f"best AUC : {val_auc}")
	return {"loss :" -val_auc, "status": STATUS_OK, "model": model}


## Searching

best_run, best_model = optim.minimize(model = create_model,
									data = data,
									algo = tpe.suggest, 
									max_evals = 100,
									trials = Trials())
print(best_model.to_json())