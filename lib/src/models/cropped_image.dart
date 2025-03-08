import 'dart:ui';

import 'package:image/image.dart' as img;

class CroppedImage {
  final img.Image image;
  final Rect location;

  const CroppedImage(
    this.image,
    this.location,
  );
}
