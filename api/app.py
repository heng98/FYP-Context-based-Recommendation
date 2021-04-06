from flask import (
    Flask,
    request,
    render_template,
    jsonify
)

from api.corpus import Corpus
from api.document_embedding import DocumentEmbeddingModel
from Transformer.candidate_selector.ann.ann_annoy import ANNAnnoy
from api.candidate_selection.ann_candidate_selector import ANNCandidateSelector



app = Flask(__name__)

corpus = Corpus()
document_embedding_model = DocumentEmbeddingModel("weights/s2orc_cs/weights_0")

ann = ANNAnnoy.load("ann/s2orc_cs/weights_0/s2orc-cs-train-val.ann")
candidate_selector = ANNCandidateSelector(ann, 8, corpus)


@app.route('/', methods=["GET"])
def index():
    paper_count, citation_count = corpus.get_stats()
    paper_count = format(paper_count, ",")
    citation_count = format(citation_count, ",")
    return render_template("index.html", paper_count=paper_count, citation_count=citation_count)

@app.route('/recommend', methods=["POST"])
def recommend():
    if request.method == "POST":
        title = request.json["title"]
        abstract = request.json["abstract"]
        
        query_embedding = document_embedding_model.embed(title, abstract)
        candidate_papers = candidate_selector.get_candidate(query_embedding)
        result = [
            {
                "title": c["title"],
                "abstract": c["abstract"]
            }
            for c, _ in candidate_papers[:30]
        ]

        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)




