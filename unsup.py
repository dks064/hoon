## unsupervised learning

X = TBRLZM010

X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 42)
logger.info('total_data : {}'.format(X['UUID'].count()))
logger.info('X_train : {}'.format(X['UUID'].count()))
logger.info('X_test : {}'.format(X['UUID'].count()))

## use_col = [ ~~~~ ]
X_train = X_train[use_col]
result = pd.DataFrame(X_test["UUID"])
X_test = X_test[use_col]
logger.info('use columns : {}'.format(use_col))
logger.info('use columns count : {}'.format(len(use_col)))

## 참조 - 스모트 토멕
"""
from imblearn.combine import SMOTETomek
imblearn.__version__

sm = SMOTETomek(random_state = 42)

from collections import Counter
X_train, y_train = sm.fit_sample(X_train, y_train)
print(Counter(y.train))

"""

scalar = MinMaxScalar().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

save_file = os.path.join(RESULT_DIR , "MinMaxScaler.pkl")
with open(save_file, 'wb') as f:
	pickle.dump(scaler,f)

def create_dnn_ae_model(input_dim):
	input_sample = Input(shape = (input_dim,))

	intermediate = 400

	encoded = Dense(intermediate , activation = 'relu')(input_sample)
	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)

	encoded = Dense(2 , activation = 'linear', name = 'latent')(encoded)

	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)

	dencoded = Dense(input_dim , activation = 'linear', name = 'fc')(dencoded)

	model = Model(input_sample, decoded)

	return model

model = create_dnn_ae_model(X_train.shape[1])
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.summary(print_fn = logger.info)

hist = model.fit(X_train, X_train,
				epochs = 10,
				batch_size = BATCH_SIZE,
				shuffle = True,
				verbose = 1)

logger.info(pd.DataFrame(hist.history))

## 참조 - dropout model
"""

def create_dnn_sup_model(input_dim):
	input_sample = Input(shape = (input_dim,))
	layers = Dense(200, activation = 'relu')(input_sample)
	layers = Dropout(0.2)layers
	layers = Dense(150, activation = 'relu')(layers)
	layers = Dropout(0.2)layers
	layers = Dense(100, activation = 'relu')(layers)
	layers = Dropout(0.2)layers
	layers = Dense(50, activation = 'relu')(layers)
	layers = Dropout(0.2)layers
	layers = Dense(25, activation = 'relu')(layers)
	layers = Dropout(0.2)layers
	layers = Dense(1, activation = 'sigmoid')(layers)
	model = Model(input_sample, output)
	return model

model = create_dnn_sup_model(X_train.shape[1])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accurancy'])

hist = model.fit(X_train, y_train,
				epochs = 10,
				batch_size = BATCH_SIZE,
				shuffle = True,
				verbose = 1)
"""

x_pred = model.predict(X_test)

x_pred
X_test

assert len(x_pred) == len(X_test)
recons_err = [get_anomaly_score(test,pred)
			for test , pred in zip(X_test, x_pred)]
result = result.assign(recons_err = recons_err)
# result = result.assign(X_test = X_test)

fig, ax = plt.subplots()
# groups = result.groupby("recons_err")
ax.plot(result.index, result.recons_err, m
	arker = "o", ms = 6,
	linestyle = '',
	alpha = 0.3,
	label = "Fraud",
	color = "b")

ax.legend()
# plt.show()

plt.savefig(os.path.join(RESULT_DIR, 'result.png'))

result

model_json = model.to_json()
with open(os.path.join(RESULT_DIR, 'unsup_ae_1_model.json'), 'w') as json_file:
	json_file.write(model_json)
	logger.info("model saved")

model.save_weights(os.path.join(RESULT_DIR, 'unsup_ae_1_model_weight.h5'))
logger.info("model weight saved")

threshold_value = resul["x_pred"].quantile(0.9)
logger.info("confusion_matrix")
logger.info(confusion_matrix(result["X_test"], result["x_pred"] > threshold_value))
logger.info("threshold_value : {}".format(threshold_value))
logger.info("")

result["X_test"].value_counts()
save_file = os.path.join(RESULT_DIR, "result.csv")
result.to_csv(save_file, index = False)
logger.info("save result : {}".format(save_file))

fig, ax = plt.subplots()
groups = result.groupby("X_test")

for name, group in groups:
	ax.plot(group.index, group.x_pred, 
			marker ="o", ms = 6, linestyle = "",
			alpha = 0.3, label = "Fraud", color = "b")
ax.legend()

#plt.show()
plt.savefig(os.path.join(RESULT_DIR, 'result.png'))

sec = time.time() - start
h = int(sec//(60*60))
m = int((sec-(h*60)) // (60))
s = int((sec - (h*60*60) - (m*60)))
logger.info("total time [{}:{}:{}]".format(h,m,s))