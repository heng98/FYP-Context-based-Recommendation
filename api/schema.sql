CREATE TABLE IF NOT EXISTS Paper (
    PaperID TEXT PRIMARY KEY,
    Title TEXT NOT NULL,
    Abstract TEXT NOT NULL,
    Year INTEGER NOT NULL,
    AnnID INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Citations (
    CitingPaperID TEXT,
    CitedPaperID TEXT,
    FOREIGN KEY (CitingPaperID) REFERENCES Paper(PaperID), 
    FOREIGN KEY (CitedPaperID) REFERENCES Paper(PaperID),
    UNIQUE(CitingPaperID, CitedPaperID)
);