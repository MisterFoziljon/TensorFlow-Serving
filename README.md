## Run object detection model using tensorflow serving

## YOLOv8 modelini tensorflow saved modelga o'tkazib, uni tensorflow serving yordamida ishlatish va streamlit yordamida deploy qilish.


### **Requirements:**
```python
 pip install tensorflow[and-cuda]
 pip install streamlit
 pip install opencv-python
 pip install -q tensorflow_serving_api
 pip install grpc
```

### **Qo'llanma:**

* [keras.io](https://keras.io/examples/keras_recipes/tf_serving/)
* [github](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/tf_serving.py)
* [google colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/tf_serving.ipynb)

### **Information:**

* ``` REST api ``` - Eng kam qo'llaniladigan API modeli REST bo'lib, REST so'zi kengroq qo'llanilsa ham, API larning kichik bir qismi shu tarzda ishlab chiqilgan. Ushbu API uslubining o'ziga xos xususiyati shundaki, mijozlar boshqa ma'lumotlardan URL manzillarini yaratmaydilar - ular server tomonidan uzatilgan URL manzillaridan foydalanadilar. Brauzer shunday ishlaydi — u foydalanadigan URL-manzillarni qismlardan tuzmaydi va u foydalanadigan URL manzillarining veb-saytga xos formatlarini tushunmaydi; u serverdan olingan joriy sahifada topilgan yoki oldingi sahifalardan qadab qo'yilgan yoki foydalanuvchi tomonidan kiritilgan URL manzillarini shunchaki ko'r-ko'rona kuzatib boradi. Brauzer amalga oshiradigan URL-manzilning yagona tahlili HTTP so'rovini yuborish uchun zarur bo'lgan ma'lumotlarni ajratib olishdir va brauzer bajaradigan yagona URL-manzillar nisbiy va asosiy URL-lardan mutlaq URL yaratishdir. Agar sizning API REST API bo'lsa, mijozlaringiz hech qachon URL manzillaringiz formatini tushunishlari shart emas va bu formatlar mijozlarga berilgan API spetsifikatsiyasining bir qismi emas. REST API juda oddiy bo'lishi mumkin. REST API-lari bilan foydalanish uchun ko'plab qo'shimcha texnologiyalar ixtiro qilingan, masalan, JSON API, ODATA, HAL, Siren yoki JSON Hyper-Schema va boshqalar - lekin RESTni yaxshi bajarish uchun ulardan hech biri kerak emas.

```python
data = json.dumps(
    {"signature_name": "serving_default", "instances": batched_img.numpy().tolist()}
)
url = "http://localhost:8501/v1/models/model:predict"


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    rest_outputs = np.array(response["predictions"])
    return rest_outputs

rest_outputs = predict_rest(data, url)

print(f"REST output shape: {rest_outputs.shape}")
print(f"Predicted class: {postprocess(rest_outputs)}")
```


* ``` gRPC ``` - API uchun HTTP dan foydalanishning ikkinchi modeli gRPC tomonidan tasvirlangan. gRPC qopqoq ostida HTTP/2 dan foydalanadi, ammo HTTP API dizayneriga ta'sir qilmaydi. gRPC tomonidan yaratilgan stublar va skeletlar HTTP-ni mijoz va serverdan ham yashiradi, shuning uchun hech kim RPC tushunchalari HTTP bilan qanday bog'langanligi haqida tashvishlanmasligi kerak - ular faqat gRPC-ni o'rganishlari kerak.

Mijozning gRPC API dan foydalanish usuli bu uch bosqichni bajarishdir:

1. Qaysi protsedurani chaqirishni hal qiling;
2. Foydalanish uchun parametr qiymatlarini hisoblang (agar mavjud bo'lsa);
3. Parametr qiymatlarini o'tkazib, kod bilan yaratilgan stubdan foydalaning.

```python
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel("localhost:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

loaded_model = tf.saved_model.load(model_export_path)
input_name = list(loaded_model.signatures["serving_default"].structured_input_signature[1].keys())[0]

def predict_grpc(data, input_name, stub):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "model"
    request.model_spec.signature_name = "serving_default"
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(data.numpy().tolist()))
    result = stub.Predict(request)
    return result


grpc_outputs = predict_grpc(batched_img, input_name, stub)
grpc_outputs = np.array([grpc_outputs.outputs['predictions'].float_val])

print(f"gRPC output shape: {grpc_outputs.shape}")
print(f"Predicted class: {postprocess(grpc_outputs)}")
```

**Terminal yordamida ishga tushirish (global deployment)**

```shell
$ tmux new-session -d -s <session_name1> "tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=model --model_base_path=$MODEL_DIR"
$ tmux new-session -d -s <session_name2> "streamlit run deploy.py --server.port <port>"
$ tmux new-session -d -s <session_name3> "ngrok http <port>"
$ tmux attach -t <session_name3>
```

**Terminal yordamida ishga tushirish (local deployment)**
* Tensorflow model serverni ishga tushirib olamiz:
```shell
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=model --model_base_path=$MODEL_DIR
```
* streamlit yordamida kodni ishga tushiramiz:
```python
streamlit run deploy.py
```

Modelni yuklab olish uchun [link](https://drive.google.com/drive/folders/1lHszpAS8PqkCZjJ2wV212AUoq_LkWxR0?usp=sharing)
