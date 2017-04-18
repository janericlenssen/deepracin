#ifndef DR_PARSER_H
#define DR_PARSER_H
#include "dR_base.h"


/**
* \brief Parses a model folder and fills the net object with structure and weights (deprecated)
*
* \author jan eric lenssen
*
* \param net An empty dR_Graph object that will be filled by the Parser
* \param path Path to the model folder

*/
dR_Node* dR_parseModel(dR_Graph* net, dR_Node* input, gchar* path, dR_Node** nodelist, gint* numnodes);


/**
* \brief Parses a graph folder and fills the net object with structure and weights
*
* \author jan eric lenssen
*
* \param net An empty dR_Graph Object that will be filled by the Parser
* \param path Path to the model folder
* \param[out] nodes Contains all graph nodes after function call
* \param[out] numnodes Contains number of graph nodes after function call
* \param[out] feednodes Contains all graph feed nodes after function call
* \param[out] numfeednodes Contains number of graph feed nodes after function call

*/
dR_Node* dR_parseGraph(dR_Graph* net, gchar* path, dR_Node*** nodelist, gint* numnodes, dR_Node*** feednodes, gint* numfeednodes);


/**
* \brief Serializes a graph and writes a model folter to path, which can be loaded with dR_parseModel()
*
* \author jan eric lenssen
*
* \param net A dR_Graph Object that will be serialized and saved
* \param path Path to the model folder

*/
gboolean dR_serializeGraph(dR_Graph* net, gchar* path);

/**
* \brief Prints the net to Console or file
*
* \author jan eric lenssen
*
* \param net A dR_Graph Object to print
* \param path Path to file where the net should be printed. If NULL, it will be printed to console.
* \param printVariables Flag for printing stored Variables (Weights and Biases).

*/
void dR_printNet(dR_Graph* net, char* path);


void dR_printSchedule(dR_Graph* net);

#endif // DR_PARSER_H
