import sqlite3
from contextlib import closing

connection = sqlite3.connect("s2orc.db", check_same_thread=False)
# connection.executescript(open("api/schema.sql").read())
# connection.commit()


class Corpus:
    def get_paper_by_id(self, paper_id):
        query = """
        SELECT Title, Abstract, Year
        FROM Paper
        WHERE PaperID = ?
        """

        with closing(connection.cursor()) as c:
            c.execute(query, (paper_id,))
            paper = c.fetchone()

        return {"title": paper[0], "abstract": paper[1], "year": paper[2]}

    def get_paper_by_ids_list(self, paper_ids_list):
        query = f"""
        SELECT Title, Abstract, Year, AnnID
        FROM Paper
        WHERE PaperID in ({",".join(["?"] * len(paper_ids_list))})
        """

        with closing(connection.cursor()) as c:
            c.execute(query, paper_ids_list)
            papers = c.fetchall()

        return [
            {
                "title": paper[0],
                "abstract": paper[1],
                "year": paper[2],
                "ann_id": paper[3],
            }
            for paper in papers
        ]

    def get_paper_by_ann_id(self, ann_id):
        query = """
        SELECT Title, Abstract, Year
        FROM Paper
        WHERE AnnID = ?
        """

        with closing(connection.cursor()) as c:
            c.execute(query, (ann_id,))
            paper = c.fetchone()

        return {"title": paper[0], "abstract": paper[1], "year": paper[2]}

    def get_paper_ids_by_ann_ids_list(self, ann_ids_list):
        query = f"""
        SELECT PaperID
        FROM Paper
        WHERE AnnID in ({",".join(["?"] * len(ann_ids_list))})
        """

        with closing(connection.cursor()) as c:
            c.execute(query, ann_ids_list)
            query_result = c.fetchall()

        paper_ids_list = [r[0] for r in query_result]

        return paper_ids_list

    def get_citation_by_ids_list(self, ids_list):
        query = f"""
        SELECT CitedPaperID
        FROM Citations
        WHERE CitingPaperID IN ({",".join(["?"] * len(ids_list))})
        """

        with closing(connection.cursor()) as c:
            c.execute(query, (*ids_list,))
            query_result = c.fetchall()

        paper_ids_list = [r[0] for r in query_result]

        return paper_ids_list

    def get_stats(self):
        query_paper = """
        SELECT COUNT(*)
        FROM Paper
        """

        query_ciation ="""
        SELECT COUNT(*)
        FROM Citations
        """

        with closing(connection.cursor()) as c:
            c.execute(query_paper)
            paper_count = c.fetchone()

            c.execute(query_ciation)
            citation_count = c.fetchone()

        return paper_count[0], citation_count[0]