package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        List<String> vocabList = Toolkit.getListVocabulary();
        List<double[]> vectorList = Toolkit.getlistVectors();

        for (int i=0; i<Toolkit.listVocabulary.size();i++){
            for (int j=0; j<Toolkit.STOPWORDS.length;j++) {
                String vocabulary = Toolkit.listVocabulary.get(i);
                String STOPWORD = Toolkit.STOPWORDS[j];
                if(vocabulary.equals(STOPWORD)) {
                    vocabList.remove(i);
                    vectorList.remove(i);
                }
            }
            String word = vocabList.get(i);
            Vector vector = new Vector(vectorList.get(i));
            Glove newGlove = new Glove(word,vector);
            listResult.add(newGlove);
        }
        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        List<Glove> listofGloves = createGloveList();
        List<Integer> documentLengths = new ArrayList<>();
        List<ArticlesEmbedding> allDocuments = _listEmbedding;

        for(ArticlesEmbedding embedding: _listEmbedding){
           String postTextProcessed = embedding.getNewsContent();
           String[] indivWords = postTextProcessed.split(" ");

           int lengthIEwordsInDocCount = 0 ;

           for (int i=0; i<indivWords.length;i++){
               String potentialWord = indivWords[i];
               for(int j=0; j<listofGloves.size();j++){
                   if (listofGloves.get(j).getVocabulary().equals(potentialWord)) {
                       lengthIEwordsInDocCount = lengthIEwordsInDocCount + 1;
                       break;
                   }
               }
           }
           documentLengths.add(lengthIEwordsInDocCount);
        }

        int n = documentLengths.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (documentLengths.get(j) > documentLengths.get(j + 1)) {
                    int temp = documentLengths.get(j);
                    documentLengths.set(j, documentLengths.get(j + 1));
                    documentLengths.set(j + 1, temp);
                }
            }
        }

        int size = documentLengths.size();
        if (size%2 ==0){
            intMedian = (documentLengths.get((size/2)+1) + documentLengths.get((size/2)))/2;
        } else{
            intMedian = documentLengths.get(((size+1)/2));
        }
        return intMedian;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for(ArticlesEmbedding anEmbedding : listEmbedding){
            try{
                anEmbedding.getEmbedding();

            }catch(InvalidSizeException e){
                anEmbedding.setEmbeddingSize(embeddingSize);
            }catch(InvalidTextException e){
                String processedText = anEmbedding.getNewsContent();
            }catch(Exception e){
                System.out.println(e.getMessage());
            }

        }

    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks

        for(ArticlesEmbedding anEmbedding : listEmbedding){

            NewsArticles.DataType newsType = anEmbedding.getNewsType();
            String label = anEmbedding.getNewsLabel();

            if (newsType.equals(NewsArticles.DataType.Training)){
                inputNDArray = anEmbedding.getEmbedding();
                outputNDArray = Nd4j.zeros(1,_numberOfClasses);

                for(int j=0;j< _numberOfClasses;j++){
                    if (Integer.parseInt(label)-1 == j){
                        outputNDArray.putScalar(new int[]{0,j},1);
                    }
                 }
                }

                DataSet myDataSet = new DataSet(inputNDArray,outputNDArray);
                listDS.add(myDataSet);
            }

        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks

        for(int i=0;i<_listEmbedding.size();i++){
            ArticlesEmbedding anArticle = _listEmbedding.get(i);

            if(anArticle.getNewsType().equals(NewsArticles.DataType.Testing)){
                int[] predictionArray = myNeuralNetwork.predict(anArticle.getEmbedding());
                listResult.add(predictionArray[0]);

                anArticle.setNewsLabel(String.valueOf(predictionArray[0]));
            }
        }

        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks

        int numberOfGroups = -1;
        List<String> groupedArticles = new ArrayList<>();

        for (ArticlesEmbedding anEmbedding : listEmbedding) {
            if (anEmbedding.getNewsType() == NewsArticles.DataType.Testing) {
                if (Integer.parseInt(anEmbedding.getNewsLabel()) > numberOfGroups) {
                    numberOfGroups = Integer.parseInt(anEmbedding.getNewsLabel());
                }
            }
        }

        for (int i=0; i<=numberOfGroups; i++){
            for (ArticlesEmbedding anEmbedding: listEmbedding){
                if (anEmbedding.getNewsType() == NewsArticles.DataType.Testing){
                    if (Integer.parseInt(anEmbedding.getNewsLabel())==i){
                        groupedArticles.add(anEmbedding.getNewsTitle());
                    }
                }
            }

            int groupNumber=i+1;
            System.out.println("Group "+ groupNumber);
            for (String oneOfTheArticles:groupedArticles){
                System.out.println(oneOfTheArticles);
            }
            groupedArticles.clear();
        }

    }



}
