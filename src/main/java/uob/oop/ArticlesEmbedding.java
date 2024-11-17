package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);

    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
    intSize = _size;

    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if (processedText == "") {
            processedText = textCleaning(super.getNewsContent());

            // set up pipeline properties
            Properties props = new Properties();
            // set the list of annotators to run
            props.setProperty("annotators", "tokenize,pos,lemma");
            // build pipeline
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            // create a document object
            CoreDocument document = pipeline.processToCoreDocument(processedText);
            // display tokens
            StringBuilder lemmatizedText = new StringBuilder();
            for (CoreLabel tok : document.tokens()) {
                lemmatizedText.append(tok.lemma()).append(" ");
            }
            processedText = lemmatizedText.toString().toLowerCase();

            StringBuilder mySB = new StringBuilder();

            String[] contentWordList = processedText.split(" ");
            for (String word : contentWordList) {

                boolean isStopWord = false;
                for (String stopWord : Toolkit.STOPWORDS) {
                    if (stopWord.equals(word)) {
                        isStopWord = true;
                        break;
                    }
                }
                if (!isStopWord) {
                    mySB.append(word).append(" ");
                }
            }
            processedText = mySB.toString();

        }
        return processedText.trim();
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
            try {
               if(!newsEmbedding.isEmpty()){
                    return newsEmbedding;
                }

                intSize = getEmbeddingSize();
                int sizeOfVector = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize();

                if (intSize == -1) {
                    throw new InvalidSizeException("Invalid size");
                }
                if (processedText.isEmpty()) {
                    throw new InvalidTextException("Invalid text");
                }

                String[] splitUpWords = processedText.split(" ");
                newsEmbedding = Nd4j.zeros(intSize, sizeOfVector);
                int row = 0;
                for (String indivWord : splitUpWords) {
                    boolean isPartOf = false;
                    for (Glove glove : AdvancedNewsClassifier.listGlove) {
                        if (glove.getVocabulary().equals(indivWord)) {
                            newsEmbedding.putRow(row, Nd4j.create(glove.getVector().getAllElements()));
                            isPartOf = true;
                            row = row + 1;
                            break;
                        }
                    }
                    if (!isPartOf) {
                        newsEmbedding.putRow(row, Nd4j.zeros((sizeOfVector)));
                    }
                    if (row >= intSize) {
                        break;
                    }
                }
            } catch (InvalidSizeException | InvalidTextException Exception) {
                if (Exception instanceof InvalidSizeException) {
                    throw new InvalidSizeException("Invalid size");
                } else {
                    throw new InvalidTextException("Invalid text");
                }

            }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
