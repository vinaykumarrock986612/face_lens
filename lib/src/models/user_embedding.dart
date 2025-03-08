class UserEmbedding {
  final int? userId;
  final String embeddings;

  const UserEmbedding({
    this.userId,
    required this.embeddings,
  });

  List<double> get embeddingList => embeddings.split(',').map((e) => double.parse(e)).toList();
}
