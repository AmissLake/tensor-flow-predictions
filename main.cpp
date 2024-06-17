#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/saved_model/loader.h>

using namespace cv;

// Função para pré-processar a imagem
void preprocess_image(const std::string& image_path, tensorflow::Tensor& input_tensor) {
    cv::Mat image = cv::imread(image_path);
    cv::resize(image, image, cv::Size(299, 299));
    image.convertTo(image, CV_32F);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image /= 255.0f;
    tensorflow::Tensor input_tensor_temp(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, image.rows, image.cols, 3}));
    auto input_tensor_mapped = input_tensor_temp.tensor<float, 4>();
    for (int y = 0; y < image.rows; ++y) {
        const float* source_row = image.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            const float* source_pixel = source_row + (x * image.channels());
            float* dest_pixel = input_tensor_mapped(0, y, x, 0);
            dest_pixel[0] = source_pixel[0];
            dest_pixel[1] = source_pixel[1];
            dest_pixel[2] = source_pixel[2];
        }
    }
    input_tensor = input_tensor_temp;
}

// Função para exibir as previsões
void display_predictions(const std::vector<std::pair<std::string, float>>& predictions) {
    for (int i = 0; i < 3; ++i) {
        std::cout << i + 1 << ". " << predictions[i].first << ": " << (predictions[i].second * 100) << "%\n";
    }
}

int main() {
    // Carregar o modelo pré-treinado InceptionV3
    tensorflow::Session* session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << "Erro ao criar a sessão TensorFlow: " << status.ToString() << std::endl;
        return 1;
    }
    status = tensorflow::LoadSavedModel(session, tensorflow::RunOptions(), "inception_v3_saved_model", {"serve"}, nullptr);
    if (!status.ok()) {
        std::cerr << "Erro ao carregar o modelo TensorFlow: " << status.ToString() << std::endl;
        return 1;
    }

    // Caminho para a imagem a ser reconhecida
    std::string image_path = "caminho/para/sua/imagem.jpg";

    // Preprocessar a imagem
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 299, 299, 3}));
    preprocess_image(image_path, input_tensor);

    // Fazer a predição usando o modelo InceptionV3
    std::vector<std::pair<std::string, float>> predictions;
    tensorflow::Tensor output_tensor;
    status = session->Run({{"input_1:0", input_tensor}}, {"dense_2/Softmax:0"}, {}, &output_tensor);
    if (!status.ok()) {
        std::cerr << "Erro ao fazer a predição: " << status.ToString() << std::endl;
        return 1;
    }
    auto output_tensor_mapped = output_tensor.tensor<float, 2>();
    for (int i = 0; i < 3; ++i) {
        predictions.emplace_back("label", output_tensor_mapped(0, i));
    }

    // Imprimir as previsões
    display_predictions(predictions);

    // Carregar e exibir a imagem com OpenCV
    cv::Mat img = cv::imread(image_path);
    cv::imshow("Image", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
