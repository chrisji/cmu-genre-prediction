import json


class CMUBookDataset:
    """
    Wrapper for the CMU Book Dataset, providing convenient functions for data access.
    """
    def __init__(self):
        self.dataset = dict()
        self.genre_map = dict()

        # Construct the dataset from the raw
        with open('../data/booksummaries.txt', 'r') as fp:
            for line in fp.readlines():
                entry = CMUBookDataset.process_book_entry(line)
                self.dataset[entry['wiki_id']] = entry  # Add book entry using wiki_id as the key.

    def all_summaries(self):
        """
        Generator for obtaining all summaries in no particular order.

        Yields a single summary string at a time, in no particular order.
        """
        for entry in self.dataset.itervalues():
            yield entry['summary']

    def all_genres(self):
        """
        Returns a set of all genres seen in the dataset.

        :return: a set of all genres seen in the dataset.
        """
        genre_set = set()

        for entry in self.dataset.itervalues():
            for genre in entry['genres']:
                genre_set.add(genre)

        return genre_set

    def get_genres(self, wiki_id, id_form=False):
        """
        Return a list of genres for a given book id (wiki id format), or None if no genre exists.

        :param wiki_id: book id to retrive genre list for
        :param id_form: False to return genre strings, True to return genre integer ids
        :return:
        """
        if id_form:
            ids = list()
            for genre in self.dataset[wiki_id]['genres']:
                ids.append(self.genre_map[genre])
            return ids
        return self.dataset[wiki_id]['genres']

    def get_summary_by_id(self, wiki_id):
        return self.dataset[wiki_id]['summary']

    def get_summary_by_title(self, title):
        for entry in self.dataset.itervalues():
            if title.lower().strip() in entry['title'].lower():
                return entry['summary']
        return None

    @staticmethod
    def process_book_entry(line):
        components = line.split('\t')

        return dict(
            wiki_id=components[0].strip(),
            freebase_id=components[1].strip(),
            title=components[2].strip(),
            author=components[3].strip(),
            publication_date=components[4].strip(),
            genres=CMUBookDataset.process_genres(components[5]),
            summary=components[6].strip()
        )

    @staticmethod
    def process_genres(genre_string):
        if len(genre_string.strip()) == 0:
            return None
        else:
            return [g.lower() for g in json.loads(genre_string).values()]

if __name__ == '__main__':
    ds = CMUBookDataset()
    print ds.get_summary_by_title('ClockWork or')