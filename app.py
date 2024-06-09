from flask import Flask, render_template, request
from keras.src.saving.saving_api import load_model
import autoreset
import preprocessData as pp
import Model as models
import vae

#creating an instance (Flask)
app=Flask(__name__)
obj= pp.preprocess_data()
dir_path="train"
#obj.visualize_images(dir_path,nimages=3)
leaf_df,train,labels=obj.preprocess(dir_path)
tr_gen,tt_gen,va_gen=obj.generate_train_test_split(leaf_df,train,labels)
input_shape=(128,128,3)
ms=models.DeepANN()

#render/route    -Normal route
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data.html")
def data():
    return render_template("data.html")

@app.route("/classes.html")
def classes():
    return render_template("classes.html")

@app.route("/modell.html")
def modell():
    return render_template("modell.html")

@app.route("/index.html")
def sample():
    return render_template("index.html")

@app.route('/model/simple_ANN0')
def model_function0():
    sann = []
    sann.append(ms.simple_ANN(input_shape=input_shape, optimizer="adam"))
    models.compare_model1(sann, ['ann_adam'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html', model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/simple_ANN')
def model_function1():
    sann=[]
    sann.append(ms.simple_ANN(input_shape=input_shape, optimizer="adam"))
    sann.append(ms.simple_ANN(input_shape=input_shape, optimizer="sgd"))
    sann.append(ms.simple_ANN(input_shape=input_shape, optimizer="rmsprop"))
    models.compare_model1(sann,['ann_adam','ann_sgd','ann_rmsprop'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/cnn_model')
def model_function2():
    cn=[]
    cn.append(ms.cnn_model(input_shape=input_shape, optimizer="adam"))
    cn.append(ms.cnn_model(input_shape=input_shape, optimizer="sgd"))
    cn.append(ms.cnn_model(input_shape=input_shape, optimizer="rmsprop"))
    models.compare_model1(cn,['cnn_adam','cnn_sgd','cnn_rmsprop'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/cnn_add_regularize')
def model_function3():
    cn=[]
    cn.append(ms.cnn_model(input_shape=input_shape, optimizer="adam"))
    models.compare_model1(cn,['cnn_adam'] , tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html', model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/cnn_vgg')
def model_function4():
    vg=[]
    vg.append(ms.cnn_vgg())
    models.compare_model1(vg,['VGG'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/rnn')
def model_function5():
    rn=[]
    rn.append(ms.rnn(input_shape=input_shape,num_class=3))
    models.compare_model1(rn,['RNN'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/model/LSTM_model')
def model_function6():
    lstm=[]
    lstm.append(ms.LSTM_model(input_shape=input_shape,num_class=3))
    models.compare_model1(lstm,['LSTM'], tr_gen, tt_gen, va_gen, 3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/autoencoder')
def model_function7():
    def autoencoder_call():
        autoen = models.Autoencoder()
        autoencoder_model, train_gen, test_gen = autoen.anmodel()
        autoencoder_model.fit(train_gen, epochs=50, validation_data=test_gen)
        autoencoder_model.save('autoencoder_saved.h5')
        loss = autoencoder_model.evaluate(train_gen)
        print("Test loss:", loss)
        model = load_model('autoencoder_saved.h5')
        nimages = 10
        test_images, _ = next(train_gen)
        predicted_images = autoencoder_model.predict(test_images)
        obj.display_images(test_images, predicted_images, 5)
    autoencoder_call()
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/autoencoderresnet')
def model_function8():
    autoreset
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/vae')
def model_function9():
    vae
    return render_template('index.html', model_comparison_graph_url='/static/images/compare.jpg')

@app.route('/compare_models')
def compare_models():
    mss = []
    mss.append(ms.simple_ANN(input_shape=input_shape, optimizer="adam"))
    mss.append(ms.simple_ANN(input_shape=input_shape, optimizer="sgd"))
    mss.append(ms.simple_ANN(input_shape=input_shape, optimizer="rmsprop"))
    mss.append(ms.cnn_model(input_shape=input_shape, optimizer="adam"))
    mss.append(ms.cnn_model(input_shape=input_shape, optimizer="sgd"))
    mss.append(ms.cnn_model(input_shape=input_shape, optimizer="rmsprop"))
    mss.append(ms.cnn_add_regularize())
    mss.append(ms.cnn_vgg())
    mss.append(ms.rnn(input_shape=input_shape, num_class=3))
    mss.append(ms.LSTM_model(input_shape=input_shape, num_class=3))
    models.compare_model1(mss,['ann_adam','ann_sgd','ann_rmsprop','cnn_adam','cnn_sgd','cnn_rmsprop','cnnregularization','vgg','rnn''lstm'],tr_gen, tt_gen, va_gen,3)
    return render_template('index.html',  model_comparison_graph_url='/static/images/compare.jpg')

if __name__=="__main__":
    app.run(debug=True)