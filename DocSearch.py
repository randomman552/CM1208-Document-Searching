import numpy as np
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
import json


def read_file(filepath: str) -> str:
    """
    read_file function reads in the given file and returns a string containing the files contents.\n
    @param filepath - The path of the file to read.
    @return - String containing the contents of the file
    """

    # Assign default return value
    retval = ""

    # Try to open file and read contents
    try:
        with open(filepath, "r") as file:
            retval = file.read()
    except FileNotFoundError as e:
        print(e)
    finally:
        # Return string of file
        return retval


def build_dict(docs: str) -> OrderedDict:
    """
    Build a dict of all unique words in the corpus.\n
    @param docs - The documents string read from the docs.txt file. Each line is assumed to be a document.
    @return - The dict of all words in the corpus, with their associated count
    """

    # Initalise empty dict for storing return value
    output = OrderedDict()

    # Split each document into its own string using splitlines
    docs = docs.splitlines()

    for doc in docs:
        # Create list of words by splitting it
        words = doc.split()
        for word in words:
            # If the word is already present, increase the stored counter, otherwise add it do the dict
            if word in output:
                output[word] += 1
            else:
                output[word] = 1

    return output


def build_index_for(word_list:list, docs:list) -> dict:
    """Builds the inverted index for just one word.\n
    This is intended to be called by a multiprocessing.Pool object.\n
    @param word - The word to be searched for.\n
    @param docs - The list of documents to be searched through (should be a list of strings).\n
    @return - An inverted index dict for the given word list"""
    
    #Build the inverted index dict
    inverted_index = dict()
    for word in word_list:
        #Create empty list for storing document ID's
        inverted_index[word] = []
        #For each document, check if the word is present. If it is, add that document's ID to the index.
        for i in range(len(docs)):
            if word in docs[i]:
                inverted_index[word].append(i)

    return inverted_index

def build_inverted_index(docs: str) -> OrderedDict:
    """
    Build the inverted index for each doc in the corps\n
    @param docs - The documents string read from the docs.txt file. Each line is assumed to be a document.
    """

    words_dict = build_dict(docs)
    docs = docs.splitlines()

    # Initalise inverted_index as an empty dict
    inverted_index = OrderedDict()

    # Create a multiprocessing pool for making the index
    pool = Pool()
    cpu_num = cpu_count()

    # Build list of words
    word_list = []
    for word in words_dict:
        word_list.append(word)

    # Split the inverted index into a number of separte lists of words to process on different processes using a Pool
    #REsults objects are stored in this results list
    results = []
    for i in range(cpu_num):
        #To ensure we get all words contained in the processes, start_index is rounded down, and end_index is rounded up
        start_index = round((i * (len(word_list) / cpu_num)) - 0.5)
        end_index = round((start_index + (len(word_list) / cpu_num)) + 0.5)
        process_word_list = word_list[start_index:end_index]
        results.append(pool.apply_async(build_index_for, args=(process_word_list, docs)))

    #For each result, call .get() on it to wait for the results
    for result in results:
        partial_index = result.get()
        #Copy the data from the generated partial index, and put it in the full inverted index
        for word in partial_index:
            inverted_index[word] = partial_index[word]
    
    #Return our result
    return inverted_index


def process_query(query: str, inverted_index: dict, docs:str) -> None:
    """
    Process a given query and print the results.\n
    1 - Print the query 'Query: {query}'\n
    2 - Print a list of the relevant documents in form '{doc1} {doc2} {etc}'\n
    3 - For each document in the list, print '{docID} {Angle compared to the search query}'\n
    @param query - The query to be searched: should be str.\n
    @param inverted_index - A dict contain the inverted index for the corpus.\n
    @param docs - The documents to search through, as a string.
    """

    pass

def process_queries(queries: str, inverted_index:dict, docs:str) -> None:
    """
    Process a number of queries.\n
    @param queries - A string of queries, each query should be separted by a new line.\n
    @param inverted_index - A dict contain the inverted index for the corpus.\n
    @param docs - The documents to search through, as a string. Each document is separated by a newline
    """

    pass


# Main
if __name__ == "__main__":
    docs = read_file("set3/docs.txt")
    queries = read_file("set3/queries.txt")
    index = build_inverted_index(docs)
