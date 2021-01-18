# Tigrigna-convoluation-using-word2vec#

 ## Heading 2 ##
 Word Embeddings  
Word embedding is foundational to natural language processing and represents the words in a text in an R-dimensional vector space, thereby enabling the capture of semantics, semantic similarity between words, and syntactic information for words. Word embedding approaches via word2vec have been proposed by Mikolov et al. [15]. Pennington et al [36] and Arora et al. [41] has introduced Word2vec's semantic similarity is a standard sequence embedding method that translates natural language into distributed representations of vectors; however, in order to overcome the inability of a predefined dictionary to learn rare word representations, FastText [42] is also used for character embedding. The word2vec and FastText models, which include two separate components (CBOW and skip-gram), can capture contextual word-to-word relationships in a multidimensional space as a preliminary step for predictive models used for semantics and data retrieval tasks [14,40]. Error! Reference source not found.Figure 1 shows that when the context words are given, the CBOW component infers the target word, while the skip-gram component infers the context words when the input word is provided [43]. In addition to that, the input, projection, and output layers are available for both learning algorithms, although their processes of output formulation are different. The input layer receivesW_n={W_((c-2),) W_((c-1)  ),…..,W_((c+1),) W_((c+2) ) } as arguments, where W_n denotes words. The projection layer corresponds to an array of multidimensional vectors and stores the sum of several vectors. The output layer corresponds to the layer that outputs the results of the vectorization
