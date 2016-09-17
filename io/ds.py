import json


class BookDataset:

    def __init__(self, dataset_path='../data/booksummaries.txt'):
        self.dataset = dict()

        with open(dataset_path, 'r') as fp:
            for line in fp.readlines():
                entry = BookDataset.process_book_entry(line)
                self.dataset[entry['wiki_id']] = entry  # Add book entry using wiki_id as the key.

    def all_summaries(self):
        for entry in self.dataset.itervalues():
            yield entry['summary']

    def all_genres(self):
        genre_set = set()

        for entry in self.dataset.itervalues():
            for genre in entry['genres']:
                genre_set.add(genre)

        return list(genre_set)

    def get_genres(self, wiki_id):
        return self.dataset[wiki_id]['genres']

    def get_summary(self, wiki_id):
        return self.dataset[wiki_id]['summary']

    @staticmethod
    def process_book_entry(line):
        components = line.split('\t')

        return dict(
            wiki_id=components[0],
            freebase_id=components[1],
            title=components[2],
            author=components[3],
            publication_date=components[4],
            genres=BookDataset.process_genres(components[5]),
            summary=components[6]
        )

    @staticmethod
    def process_genres(genre_string):
        if len(genre_string.strip()) == 0:
            return None
        else:
            return json.loads(genre_string).values()

if __name__ == '__main__':
    ds = BookDataset()
    print ds.all_genres()[:10]
    print list(ds.all_summaries())[0]