%{
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <cstring>
#include <iostream>

#include "euler/parser/tree.h"
typedef struct yy_buffer_state * YY_BUFFER_STATE;
typedef size_t yy_size_t;
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern int yylex();
extern int yyparse();

void yyerror(const char* s);

TreeNode* t = NULL;

%}

%union {
  class TreeNode* node;
}

%token<node> v e sample_node sample_edge sample_n_with_types
%token<node> select_ v_select
%token<node> out_v in_v out_e sample_neighbor sample_l_nb
%token<node> values label udf
%token<node> p num l r limit order_by desc asc as or_ and_ has has_key has_label gt ge lt le eq ne
%token end

%type<node> TRAV ROOT_NODE ROOT_EDGE
%type<node> GET_VALUE_WITH_SELECT GET_VALUE SELECT_VALUE
%type<node> SEARCH_NODE_WITH_SELECT SEARCH_NODE
%type<node> SEARCH_EDGE_WITH_SELECT SEARCH_EDGE
%type<node> API_GET_NODE API_SAMPLE_NODE API_SAMPLE_N_WITH_TYPES API_SAMPLE_LNB
%type<node> API_GET_EDGE API_SAMPLE_EDGE
%type<node> API_GET_P API_GET_NODE_T
%type<node> API_GET_NB_NODE API_GET_RNB_NODE
%type<node> API_GET_NB_EDGE API_SAMPLE_NB
%type<node> SELECT V_SELECT
%type<node> V E SAMPLE_NODE SAMPLE_N_WITH_TYPES SAMPLE_EDGE SAMPLE_NB SAMPLE_LNB VA PARAMS CONDITION
%type<node> POST_PROCESS LIMIT ORDER_BY AS DNF CONJ TERM
%type<node> HAS HAS_LABEL HAS_KEY SIMPLE_CONDITION

%%

GREMLIN: TRAV end {return 0;}
;

TRAV: ROOT_NODE {t = new TreeNode("TRAV"); t->AddChild($1); $$ = t;}
  | ROOT_EDGE {t = new TreeNode("TRAV"); t->AddChild($1); $$ = t;}
  | ROOT_NODE SEARCH_NODE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(2, $1, $2); $$ = t;}
  | ROOT_NODE SEARCH_EDGE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(2, $1, $2); $$ = t;}
  | ROOT_NODE GET_VALUE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(2, $1, $2); $$ = t;}
  | ROOT_EDGE GET_VALUE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(2, $1, $2); $$ = t;}
  | ROOT_NODE SEARCH_NODE_WITH_SELECT GET_VALUE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | ROOT_NODE SEARCH_EDGE_WITH_SELECT GET_VALUE_WITH_SELECT {t = new TreeNode("TRAV"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

ROOT_NODE: API_GET_NODE {t = new TreeNode("ROOT_NODE"); t->AddChild($1); $$ = t;}
  | API_SAMPLE_NODE {t = new TreeNode("ROOT_NODE"); t->AddChild($1); $$ = t;}
  | API_SAMPLE_N_WITH_TYPES {t = new TreeNode("ROOT_NODE"); t->AddChild($1); $$ = t;}
;

ROOT_EDGE: API_GET_EDGE {t = new TreeNode("ROOT_EDGE"); t->AddChild($1); $$ = t;}
  | API_SAMPLE_EDGE {t = new TreeNode("ROOT_EDGE"); t->AddChild($1); $$ = t;}
;

GET_VALUE_WITH_SELECT: GET_VALUE {t = new TreeNode("GET_VALUE_WITH_SELECT"); t->AddChild($1); $$ = t;}
  | GET_VALUE SELECT_VALUE {t = new TreeNode("GET_VALUE_WITH_SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
  | SELECT_VALUE {t = new TreeNode("GET_VALUE_WITH_SELECT"); t->AddChild($1); $$ = t;}
;

SELECT_VALUE: V_SELECT GET_VALUE {t = new TreeNode("SELECT_VALUE"); t->AddChildren(2, $1, $2); $$ = t;}
  | V_SELECT GET_VALUE SELECT_VALUE {t = new TreeNode("SELECT_VALUE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

GET_VALUE: API_GET_P {t = new TreeNode("GET_VALUE"); t->AddChild($1); $$ = t;}
  | API_GET_NODE_T {t = new TreeNode("GET_VALUE"); t->AddChild($1); $$ = t;}
;

SEARCH_NODE_WITH_SELECT: SEARCH_NODE {t = new TreeNode("SEARCH_NODE_WITH_SELECT"); t->AddChild($1); $$ = t;}
  | SELECT SEARCH_NODE {t = new TreeNode("SEARCH_NODE_WITH_SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
  | SEARCH_NODE SEARCH_NODE_WITH_SELECT {t = new TreeNode("SEARCH_NODE_WITH_SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
  | SELECT SEARCH_NODE SEARCH_NODE_WITH_SELECT {t = new TreeNode("SEARCH_NODE_WITH_SELECT"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

SEARCH_NODE: API_GET_NB_NODE {t = new TreeNode("SEARCH_NODE"); t->AddChild($1); $$ = t;}
  | API_GET_RNB_NODE {t = new TreeNode("SEARCH_NODE"); t->AddChild($1); $$ = t;}
  | API_SAMPLE_NB {t = new TreeNode("SEARCH_NODE"); t->AddChild($1); $$ = t;}
  | API_SAMPLE_LNB {t = new TreeNode("SEARCH_NODE"); t->AddChild($1); $$ = t;}
;

SEARCH_EDGE_WITH_SELECT: SEARCH_EDGE {t = new TreeNode("SEARCH_EDGE_WITH_SELECT"); t->AddChild($1); $$ = t;}
  | SELECT SEARCH_EDGE {t = new TreeNode("SEARCH_EDGE_WITH_SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
  | SEARCH_NODE SEARCH_EDGE_WITH_SELECT {t = new TreeNode("SEARCH_EDGE_WITH_SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
  | SELECT SEARCH_NODE SEARCH_EDGE_WITH_SELECT {t = new TreeNode("SEARCH_EDGE_WITH_SELECT"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

SEARCH_EDGE: API_GET_NB_EDGE {t = new TreeNode("SEARCH_EDGE"); t->AddChild($1); $$ = t;}
;

SELECT: select_ p {t = new TreeNode("SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
;

V_SELECT: v_select p {t = new TreeNode("SELECT"); t->AddChildren(2, $1, $2); $$ = t;}
;

API_GET_NODE: V {t = new TreeNode("API_GET_NODE"); t->AddChild($1); $$ = t;}
  | V AS {t = new TreeNode("API_GET_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | V CONDITION {t = new TreeNode("API_GET_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | V CONDITION AS {t = new TreeNode("API_GET_NODE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_SAMPLE_NODE: SAMPLE_NODE {t = new TreeNode("API_SAMPLE_NODE"); t->AddChild($1); $$ = t;}
  | SAMPLE_NODE AS {t = new TreeNode("API_SAMPLE_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_NODE CONDITION {t = new TreeNode("API_SAMPLE_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_NODE CONDITION AS {t = new TreeNode("API_SAMPLE_NODE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_SAMPLE_N_WITH_TYPES: SAMPLE_N_WITH_TYPES {t = new TreeNode("API_SAMPLE_N_WITH_TYPES"); t->AddChild($1); $$ = t;}
  | SAMPLE_N_WITH_TYPES AS {t = new TreeNode("API_SAMPLE_N_WITH_TYPES"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_N_WITH_TYPES CONDITION {t = new TreeNode("API_SAMPLE_N_WITH_TYPES"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_N_WITH_TYPES CONDITION AS {t = new TreeNode("API_SAMPLE_N_WITH_TYPES"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_GET_EDGE: E {t = new TreeNode("API_GET_EDGE"); t->AddChild($1); $$ = t;}
  | E AS {t = new TreeNode("API_GET_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
  | E CONDITION {t = new TreeNode("API_GET_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
  | E CONDITION AS {t = new TreeNode("API_GET_EDGE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_SAMPLE_EDGE: SAMPLE_EDGE {t = new TreeNode("API_SAMPLE_EDGE"); t->AddChild($1); $$ = t;}
  | SAMPLE_EDGE AS {t = new TreeNode("API_SAMPLE_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_EDGE CONDITION {t = new TreeNode("API_SAMPLE_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_EDGE CONDITION AS {t = new TreeNode("API_SAMPLE_EDGE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_GET_P: VA {t = new TreeNode("API_GET_P"); t->AddChild($1); $$ = t;}
  | VA AS {t = new TreeNode("API_GET_P"); t->AddChildren(2, $1, $2); $$ = t;}
;

API_GET_NODE_T: label {t = new TreeNode("API_GET_NODE_T"); t->AddChild($1); $$ = t;}
  | label AS {t = new TreeNode("API_GET_NODE_T"); t->AddChildren(2, $1, $2); $$ = t;}
;

API_GET_NB_NODE: out_v p {t = new TreeNode("API_GET_NB_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | out_v p AS {t = new TreeNode("API_GET_NB_NODE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | out_v p CONDITION {t = new TreeNode("API_GET_NB_NODE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | out_v p CONDITION AS {t = new TreeNode("API_GET_NB_NODE"); t->AddChildren(4, $1, $2, $3, $4); $$ = t;}
;

API_GET_RNB_NODE: in_v {t = new TreeNode("API_GET_RNB_NODE"); t->AddChild($1); $$ = t;}
  | in_v AS {t = new TreeNode("API_GET_RNB_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | in_v CONDITION {t = new TreeNode("API_GET_RNB_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
  | in_v CONDITION AS {t = new TreeNode("API_GET_RNB_NODE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_GET_NB_EDGE: out_e p {t = new TreeNode("API_GET_NB_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
  | out_e p AS {t = new TreeNode("API_GET_NB_EDGE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | out_e p CONDITION {t = new TreeNode("API_GET_NB_EDGE"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | out_e p CONDITION AS {t = new TreeNode("API_GET_NB_EDGE"); t->AddChildren(4, $1, $2, $3, $4); $$ = t;}
;

API_SAMPLE_NB: SAMPLE_NB {t = new TreeNode("API_SAMPLE_NB"); t->AddChild($1); $$ = t;}
  | SAMPLE_NB AS {t = new TreeNode("API_SAMPLE_NB"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_NB CONDITION {t = new TreeNode("API_SAMPLE_NB"); t->AddChildren(2, $1, $2); $$ = t;}
  | SAMPLE_NB CONDITION AS {t = new TreeNode("API_SAMPLE_NB"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

API_SAMPLE_LNB: SAMPLE_LNB {t = new TreeNode("API_SAMPLE_LNB"); t->AddChild($1); $$ = t;}
  | SAMPLE_LNB AS {t = new TreeNode("API_SAMPLE_LNB"); t->AddChildren(2, $1, $2); $$ = t;}
;

V: v {t = new TreeNode("V"); t->AddChild($1); $$ = t;}
  | v p {t = new TreeNode("V"); t->AddChildren(2, $1, $2); $$ = t;}
;

E: e {t = new TreeNode("E"); t->AddChild($1); $$ = t;}
  | e p {t = new TreeNode("E"); t->AddChildren(2, $1, $2); $$ = t;}
;

SAMPLE_NODE: sample_node PARAMS {t = new TreeNode("SAMPLE_NODE"); t->AddChildren(2, $1, $2); $$ = t;}
;

SAMPLE_N_WITH_TYPES: sample_n_with_types PARAMS {t = new TreeNode("SAMPLE_N_WITH_TYPES"); t->AddChildren(2, $1, $2); $$ = t;}

SAMPLE_EDGE: sample_edge PARAMS {t = new TreeNode("SAMPLE_EDGE"); t->AddChildren(2, $1, $2); $$ = t;}
;

SAMPLE_NB: sample_neighbor PARAMS num {t = new TreeNode("SAMPLE_NB"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

SAMPLE_LNB: sample_l_nb PARAMS num {t = new TreeNode("SAMPLE_LNB"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

VA: values PARAMS {t = new TreeNode("VA"); t->AddChildren(2, $1, $2); $$ = t;}
  | values PARAMS udf PARAMS {t = new TreeNode("VA"); t->AddChildren(4, $1, $2, $3, $4); $$ = t;}
  | values PARAMS udf PARAMS l PARAMS r {t = new TreeNode("VA"); t->AddChildren(7, $1, $2, $3, $4, $5, $6, $7); $$ = t;}
;

PARAMS: p {t = new TreeNode("PARAMS"); t->AddChild($1); $$ = t;}
  | p PARAMS {t = new TreeNode("PARAMS"); t->AddChildren(2, $1, $2); $$ = t;}
;

CONDITION: DNF {t = new TreeNode("CONDITION"); t->AddChild($1); $$ = t;}
  | POST_PROCESS {t = new TreeNode("CONDITION"); t->AddChild($1); $$ = t;}
  | DNF POST_PROCESS {t = new TreeNode("CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
;

POST_PROCESS: ORDER_BY {t = new TreeNode("POST_PROCESS"); t->AddChild($1); $$ = t;}
  | LIMIT {t = new TreeNode("POST_PROCESS"); t->AddChild($1); $$ = t;}
  | ORDER_BY LIMIT {t = new TreeNode("POST_PROCESS"); t->AddChildren(2, $1, $2); $$ = t;}

LIMIT: limit num {t = new TreeNode("LIMIT"); t->AddChildren(2, $1, $2); $$ = t;}
;

ORDER_BY: order_by p desc {t = new TreeNode("ORDER_BY"); t->AddChildren(3, $1, $2, $3); $$ = t;}
  | order_by p asc {t = new TreeNode("ORDER_BY"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

AS: as p {t = new TreeNode("AS"); t->AddChildren(2, $1, $2); $$ = t;}
;

DNF: CONJ {t = new TreeNode("DNF"); t->AddChild($1); $$ = t;}
  | CONJ or_ DNF {t = new TreeNode("DNF"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

CONJ: TERM {t = new TreeNode("CONJ"); t->AddChild($1); $$ = t;}
  | TERM and_ CONJ {t = new TreeNode("CONJ"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

TERM: HAS {t = new TreeNode("TERM"); t->AddChild($1); $$ = t;}
  | HAS_LABEL {t = new TreeNode("TERM"); t->AddChild($1); $$ = t;}
  | HAS_KEY {t = new TreeNode("TERM"); t->AddChild($1); $$ = t;}
;

HAS: has p SIMPLE_CONDITION {t = new TreeNode("HAS"); t->AddChildren(3, $1, $2, $3); $$ = t;}
;

HAS_LABEL: has_label p {t = new TreeNode("HAS_LABEL"); t->AddChildren(2, $1, $2); $$ = t;}
;

HAS_KEY: has_key p {t = new TreeNode("HAS_KEY"); t->AddChildren(2, $1, $2); $$ = t;}
;

SIMPLE_CONDITION: gt num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | lt num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | eq num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | eq p {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | ge num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | le num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
  | ne num {t = new TreeNode("SIMPLE_CONDITION"); t->AddChildren(2, $1, $2); $$ = t;}
;
%%

Tree BuildGrammarTree(std::string gremlin) {
  gremlin += "\n";
  char* str = new char[gremlin.length() + 1];
  std::strcpy(str, gremlin.c_str());
  YY_BUFFER_STATE bufferState = yy_scan_string(str);
  yyparse();
  Tree tree(t);
  yy_delete_buffer(bufferState);
  delete[] str;
  return tree;
}

void yyerror(const char* s) {
  std::cout << "Gremlin parse error!" << std::endl;
  if (t != NULL) std::cout << "after parsing: " << t->FindLeft()->GetValue() << std::endl;
  exit(1);
}
