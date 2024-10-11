package org.example;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

public class LuceneSearchEngine {
    public static final String INDEX_DIR = "index_dir";
    public static final String FIELD_NAME = "test";
    public static final String DOC_ID_FIELD_NAME = "id";
    public static final int THREAD_POOL_SIZE = 8;
    private final Analyzer analyzer;
    private final Directory index;
    private ExecutorService executors;

    public LuceneSearchEngine() throws Exception {
        this.analyzer = new StandardAnalyzer();
        this.index = new MMapDirectory(Path.of(INDEX_DIR));
        this.executors = Executors.newFixedThreadPool(THREAD_POOL_SIZE);
    }

    public void ingest(Map<String,Number>[] documents, int[] docIds) throws Exception {
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        try (IndexWriter writer = new IndexWriter(index, config)) {
            for(int i=0; i<documents.length; i++){
                Map<String,Number> content = documents[i];
                Document doc = new Document();
                doc.add(new StoredField(DOC_ID_FIELD_NAME, docIds[i]));
                for(Map.Entry<String,Number> entry: content.entrySet()){
                    doc.add(new FeatureField(FIELD_NAME, entry.getKey(), entry.getValue().floatValue()));
                }
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    public List<Integer> search(Map<String,Number> queryTokens, int topK) throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            BooleanQuery.Builder builder = new BooleanQuery.Builder();
            for (Map.Entry<String, Number> entry : queryTokens.entrySet()) {
                builder.add(FeatureField.newLinearQuery(FIELD_NAME, entry.getKey(), entry.getValue().floatValue()), BooleanClause.Occur.SHOULD);
            }
            TopDocs results = searcher.search(builder.build(), topK);
            List<Integer> resultsList = new ArrayList();
            for (ScoreDoc scoreDoc : results.scoreDocs) {
                Document doc = searcher.doc(scoreDoc.doc);
                resultsList.add(Integer.valueOf(doc.get(DOC_ID_FIELD_NAME)));
            }
            return resultsList;
        }
    }

    // 3. Batch search with a list of queries using a thread pool
    public List<List<Integer>> batchSearch(Map<String,Number>[] queries, int topK) throws Exception {
        List<Future<List<Integer>>> futures = new ArrayList<>();
        for (Map<String, Number> queryTokens : queries) {
            futures.add(executors.submit(() -> {
                return search(queryTokens, topK);
            }));
        }

        List<List<Integer>> batchResults = new ArrayList<>();
        for (Future<List<Integer>> future : futures) {
            batchResults.add(future.get());
        }
        return batchResults;
    }

    public static void main(String[] args) throws Exception {
        LuceneSearchEngine engine = new LuceneSearchEngine();
        engine.ingest(new Map[]{Map.of("a",1f,"b",2f),Map.of("a",2f,"b",1f)}, new int[]{1,2});
        System.out.println(engine.batchSearch(new Map[]{
                Map.of("a", 1f), Map.of("b", 1f)
        }, 3));
        System.out.println(engine.batchSearch(new Map[]{
                Map.of("a", 1f), Map.of("b", 1f)
        }, 1));
    }
}