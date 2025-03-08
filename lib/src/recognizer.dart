import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'models/cropped_image.dart';
import 'models/user_embedding.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  late FaceDetector faceDetector;

  static const int height = 112;
  static const int width = 112;
  static const modelName = 'assets/ml_assets/facenet.tflite';

  List<UserEmbedding> registered = [];

  Future<void> init({
    int? numThreads,
    List<UserEmbedding> allEmbeddings = const [],
  }) async {
    registered = List.from(allEmbeddings);

    _interpreterOptions = InterpreterOptions();
    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    faceDetector = FaceDetector(options: FaceDetectorOptions());
    await loadModel();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(
        modelName,
        options: _interpreterOptions,
      );
      interpreter.allocateTensors();
      printModelInputOutputDetails(interpreter);
    } catch (e) {
      debugPrint('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  List<dynamic> imageToArray(img.Image image, {int width = 160, int height = 160}) {
    final resizedImage = img.copyResize(image, width: width, height: height);
    final pixels = resizedImage.data!;
    final Float32List reshapedArray = Float32List(width * height * 3);

    int index = 0;
    for (final pixel in pixels) {
      reshapedArray[index++] = (pixel.r - 127.5) / 127.5;
      reshapedArray[index++] = (pixel.g - 127.5) / 127.5;
      reshapedArray[index++] = (pixel.b - 127.5) / 127.5;
    }

    return reshapedArray.reshape([1, height, width, 3]);
  }

  UserEmbedding? recognize(img.Image image) {
    try {
      final outputArray = getEmbeddings(image);

      if (outputArray == null) return null;

      /// Find nearest embedding in the database
      final result = findNearest(outputArray);
      debugPrint("Recognized Face: ${result?.userId}");

      return result;
    } catch (e, stackTrace) {
      debugPrint("Error in recognize(): $e\n$stackTrace");
      return null;
    }
  }

  void printModelInputOutputDetails(Interpreter interpreter) {
    final inputTensors = interpreter.getInputTensors();
    final outputTensors = interpreter.getOutputTensors();

    for (var tensor in inputTensors) {
      debugPrint("Input Tensor: ${tensor.name}, Shape: ${tensor.shape}, Type: ${tensor.type}");
    }

    for (var tensor in outputTensors) {
      debugPrint("Output Tensor: ${tensor.name}, Shape: ${tensor.shape}, Type: ${tensor.type}");
    }
  }

  List<double>? getEmbeddings(img.Image image) {
    /// Convert image to float array
    final input = imageToArray(image);
    if (input.isEmpty) {
      return null;
    }

    /// Prepare output array
    List<List<double>> output = List.generate(1, (_) => List.filled(512, 0.0));

    /// Perform inference
    final startTime = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final elapsedTime = DateTime.now().millisecondsSinceEpoch - startTime;
    debugPrint("Time to run inference: $elapsedTime ms");

    /// Convert output to double list
    List<double> outputArray = output.first.cast<double>();

    return outputArray;
  }

  UserEmbedding? findNearest(List<double> emb) {
    if (registered.isEmpty) {
      debugPrint("Warning: No registered faces found.");
      return null;
    }

    UserEmbedding? bestMatch;
    double minDistance = 0.7;

    for (final item in registered) {
      final List<double> knownEmb = item.embeddingList;

      if (emb.length != knownEmb.length) {
        debugPrint("Error: Embedding size mismatch for ${item.userId}.");
        continue;
      }

      double distance = sqrt(
        List.generate(emb.length, (i) => pow(emb[i] - knownEmb[i], 2).toDouble()).reduce((a, b) => a + b),
      );

      if (distance < minDistance) {
        minDistance = distance;
        bestMatch = item;
      }
    }

    return bestMatch;
  }

  Future<UserEmbedding?> getMatchingFace(CroppedImage cropped) async {
    return recognize(cropped.image);
  }

  Future<CroppedImage?> getCroppedImage(XFile file) async {
    // final rotated = await removeRotation(file);
    final bytes = await file.readAsBytes();
    final image = await decodeImageFromList(bytes);
    final inputImage = InputImage.fromFilePath(file.path);
    final faces = await faceDetector.processImage(inputImage);
    for (Face face in faces) {
      Rect faceRect = face.boundingBox;
      num left = faceRect.left < 0 ? 0 : faceRect.left;
      num top = faceRect.top < 0 ? 0 : faceRect.top;
      num right = faceRect.right > image.width ? image.width - 1 : faceRect.right;
      num bottom = faceRect.bottom > image.height ? image.height - 1 : faceRect.bottom;
      num width = right - left;
      num height = bottom - top;
      img.Image? faceImg = img.decodeImage(bytes);
      img.Image croppedFace = img.copyCrop(
        faceImg!,
        x: left.toInt(),
        y: top.toInt(),
        width: width.toInt(),
        height: height.toInt(),
      );
      return CroppedImage(croppedFace, faceRect);
    }
    return null;
  }

  Future<File> removeRotation(XFile image) async {
    final img.Image? capturedImage = img.decodeImage(await File(image.path).readAsBytes());
    final img.Image orientedImage = img.bakeOrientation(capturedImage!);
    return File(image.path).writeAsBytes(img.encodeJpg(orientedImage));
  }

  void close() {
    interpreter.close();
  }
}
