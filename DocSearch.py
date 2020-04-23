import numpy as np
import math
from multiprocessing import Pool, cpu_count
import json
import time


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


def build_doc_dict(doc: list) -> dict:
    """
    Create a dict containing the count of each word in the given document.\n
    @param doc - The doc to create a dict for (as a list)
    @return - A dict containing the count of each word in the given document
    """

    output = dict()

    # Go through the document by word
    for word in doc:
        # If the word has already been counted, ignore it
        if not(word in output):
            output[word] = doc.count(word)

    return output


def build_corpus_dict(docs: list) -> dict:
    """
    Should be passed the list of doc dicts.\n
    Will combine them and return a dictionary of the occurances of words in the whole corpus\n
    @param docs - List of dicts (as generated by convert_docs)\n
    @return - A dict containg all the data from all doc dicts.
    """

    # Initalise empty dict for storing return value
    output = dict()

    for doc in docs:
        # Merge the dicts
        for word in doc:
            if word in output:
                output[word] += doc[word]
            else:
                output[word] = doc[word]

    return output


def build_index_for(word_list: list, docs: list) -> dict:
    """Builds the inverted index for a list of worlds\n
    This is intended to be called by a multiprocessing.Pool object.\n
    @param word_list - The list of words to be built for\n
    @param docs - The list of documents to be searched through (should be a list of strings).\n
    @return - An inverted index dict for the given word list"""

    # Build the inverted index dict
    inverted_index = dict()
    for word in word_list:
        # Create empty list for storing document ID's
        inverted_index[word] = []
        # For each document, check if the word is present. If it is, add that document's ID to the index.
        for i in range(len(docs)):
            doc = docs[i]
            if word in doc:
                # Append the ID to the index
                inverted_index[word].append(i)

    return inverted_index


def build_inverted_index(docs: list, corpus_dict: dict) -> dict:
    """
    Build the inverted index for each doc in the corps\n
    @param docs - The documents string read from the docs.txt file. Each line is assumed to be a document.
    @param corpus_dict - The dict for all words in the corpus
    """

    # Initalise inverted_index as an empty dict
    inverted_index = dict()

    # Create a multiprocessing pool for making the index
    pool = Pool()
    cpu_num = cpu_count()

    # Build list of words
    word_list = []
    for word in corpus_dict:
        word_list.append(word)

    # Split the inverted index into a number of separte lists of words to process on different processes using a Pool
    # Results objects are stored in this results list
    results = []
    for i in range(cpu_num):
        # To ensure we get all words contained in the processes, start_index is rounded down, and end_index is rounded up
        start_index = round((i * (len(word_list) / cpu_num)) - 0.5)
        end_index = round((start_index + (len(word_list) / cpu_num)) + 0.5)
        process_word_list = word_list[start_index:end_index]
        results.append(pool.apply_async(
            build_index_for, args=(process_word_list, docs)))

    # For each result, call .get() on it to wait for the results
    for result in results:
        partial_index = result.get()
        # Copy the data from the generated partial index, and put it in the full inverted index
        for word in partial_index:
            inverted_index[word] = partial_index[word]

    # Close multithreading pool
    pool.close()

    # Return our result
    return inverted_index

# This function would be a local one within process query, but due to the use of multithreading it HAS to be outside a local scope


def build_vector(doc: dict, inverted_index: dict) -> np.array:
    """Build the vector for the given document.\n
    @param doc - The document to build the array for.\n
    @param inverted_index - The index to build it from."""

    temp_list = []

    # For each word in the index, append the count of that word in the document to the list
    for word in inverted_index:
        try:
            temp_list.append(doc[word])
        except KeyError:
            temp_list.append(0)

    # Turn the list into a vector
    vector = np.array(temp_list)
    return vector

# This function would be a local one within process query, but due to the use of multithreading it HAS to be outside a local scope


def calc_angles(relevant_docs: list, docs: list, inverted_index: dict, query: str) -> dict:
    """Calculate the angles between the query and the given docs.\n
    @param relevant_docs - A list of relevant document IDs\n
    @param docs - The documents to use (list of srtings)\n
    @param inverted_index - The inverted index for this corpus.\n
    @param query - The query to compare to.
    """

    # Build the vector for the search query
    query_vector = build_vector(build_doc_dict(query.split()), inverted_index)
    query_norm = np.linalg.norm(query_vector)

    # Create a dict to store the angles in
    angle_dict = dict()

    # For each document, print its ID and the angle between it and the search query
    for docID in relevant_docs:
        vector = build_vector(docs[docID], inverted_index)
        dot_product = np.dot(vector, query_vector)
        norm = np.linalg.norm(vector)

        # Calculate angle in radians
        angle = np.arccos(dot_product / (norm * query_norm))

        # Convert to degrees
        angle = math.degrees(angle)

        # Append the angle to the list
        angle_dict[docID] = angle

    # Return the results
    return angle_dict


def process_query(query: str, inverted_index: dict, docs: list, pool: Pool) -> None:
    """
    Process a given query and print the results. This is intended to be called via process_queries but can be called standalone assuming you create a Pool object for it\n
    1 - Print the query 'Query: {query}'\n
    2 - Print a list of the relevant documents in form '{doc1} {doc2} {etc}'\n
    3 - For each document in the list, print '{docID} {Angle compared to the search query}'\n
    @param query - The query to be searched: should be str.\n
    @param inverted_index - A dict contain the inverted index for the corpus.\n
    @param docs - The documents to search through, as a list of strings\n
    @param pool - A multhprocessing.Pool object, used to process angle calculation in parallel.
    """
    print(f"Query: '{query}'")

    # Split the query into a list of words to prevent crossover when using in operator
    query_split = query.split()

    # Use a set to store the relevant documents, to prevent any duplicates
    # Starting set contains the ID of all documents in the docs list
    relevant_docs = set([i for i in range(len(docs))])

    # Attempt to intersection the current relevant_docs set with the set of the inverted index for the current word
    # If it is not present in the inverted index, catch the raised NameError
    for word in query_split:
        try:
            relevant_docs.intersection_update(inverted_index[word])
        except NameError:
            pass

    # Print the list or relevant documents in the right format
    print("Relevant documents: '", end="")
    # This line unpacks all the items from the list and prints them separated by whitespace.
    print(*relevant_docs, sep=" ", end="'\n")

    # Initalise empty dict to store angles
    angle_dict = dict()

    # Results objects are stored in this results list
    results = []

    # Relevant docs must be converted to a list to allow for indexing for this step
    relevant_docs = list(relevant_docs)

    # Get cpu number
    cpu_num = cpu_count()

    for i in range(cpu_num):
        # To ensure we get all words contained in the processes, start_index is rounded down, and end_index is rounded up
        start_index = round((i * (len(relevant_docs) / cpu_num)) - 0.5)
        end_index = round((start_index + (len(relevant_docs) / cpu_num)) + 0.5)

        # Create the smaller doc_list as a sub-list of the main relevant docs list
        doc_list = relevant_docs[start_index:end_index]

        # Call the pool for each documents sub-list
        results.append(pool.apply_async(calc_angles, args=(
            doc_list, docs, inverted_index, query)))

    # For each result, call .get() on it to wait for the results
    for result in results:
        angle_return = result.get()

        # Copy the data from each angle_return dict into the main one
        for docID in angle_return:
            angle_dict[docID] = angle_return[docID]

    # Print the document results
    # The dict is sorted so the most relevant docIDs are printed first
    for docID in sorted(angle_dict, key=angle_dict.__getitem__):
        print(f"{docID} {format(angle_dict[docID], '.5f')}")


def process_queries(queries: str, inverted_index: dict, docs: str) -> None:
    """
    Process a number of queries.\n
    @param queries - A string of queries, each query should be separted by a new line.\n
    @param inverted_index - A dict contain the inverted index for the corpus.\n
    @param docs - The documents to search through, as a string. Each document is separated by a newline\n
    @param pool - A multithreading pool to carry out calculations on.
    """

    # Define the pool object up here to reduce the overhead from constantly creating a new object
    pool = Pool()

    # Split the queries and docs by newline character
    queries = queries.splitlines()

    # Process each query serially
    for query in queries:
        process_query(query, inverted_index, docs, pool)

    # Close the pool object
    pool.close()


def build_doc_dicts(docs: str) -> list:
    """
    Converts the string of docs into a list of dicts. The dict contains the number of occurances of each word in the dict.
    @param docs - The string to convert
    """

    # Split documents by line
    docs = docs.splitlines()

    for i in range(len(docs)):
        docs[i] = build_doc_dict(docs[i].split())

    # Add an empty document to the 0th element, so that the document list is indexed from 1.
    # The empty starting dict at 0 has no impact on the queries as it won't be present in the inverted index anyway
    docs = [dict()] + docs

    return docs


# Main
if __name__ == "__main__":
    # By default this program reads the docs.txt and queries.txt file in the same directory as it.
    docs = read_file("docs.txt")
    queries = read_file("queries.txt")
    docs = build_doc_dicts(docs)
    corpus_dict = build_corpus_dict(docs)
    index = build_inverted_index(docs, corpus_dict)
    print(f"Words in dictionary: {len(index)}")
    process_queries(queries, index, docs)
