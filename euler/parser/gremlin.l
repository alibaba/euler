%option noyywrap

%{
#include <stdio.h>
#include <string>

#include "euler/parser/gremlin.tab.h"
#include "euler/parser/tree.h"
%}

%%

[ \t\(\)\.\,] ;
\n {return end;}
"v" {yylval.node = new TreeNode("v"); return v;}
"e" {yylval.node = new TreeNode("e"); return e;}
"select" {yylval.node = new TreeNode("select"); return select_;}
"v_select" {yylval.node = new TreeNode("v_select"); return v_select;}
"outV" {yylval.node = new TreeNode("out_v"); return out_v;}
"inV" {yylval.node = new TreeNode("in_v"); return in_v;}
"outE" {yylval.node = new TreeNode("out_e"); return out_e;}
"values" {yylval.node = new TreeNode("values"); return values;}
"label" {yylval.node = new TreeNode("label"); return label;}
"sampleN" {yylval.node = new TreeNode("sample_node"); return sample_node;}
"sampleNWithTypes" {yylval.node = new TreeNode("sample_n_with_types"); return sample_n_with_types;}
"sampleE" {yylval.node = new TreeNode("sample_edge"); return sample_edge;}
"sampleNB" {yylval.node = new TreeNode("sample_neighbor"); return sample_neighbor;}
"sampleLNB" {yylval.node = new TreeNode("sample_l_nb"); return sample_l_nb;}
"limit" {yylval.node = new TreeNode("limit"); return limit;}
"order_by" {yylval.node = new TreeNode("order_by"); return order_by;}
"desc" {yylval.node = new TreeNode("desc"); return desc;}
"asc" {yylval.node = new TreeNode("asc"); return asc;}
"as" {yylval.node = new TreeNode("as"); return as;}
"or" {yylval.node = new TreeNode("or"); return or_;}
"and" {yylval.node = new TreeNode("and"); return and_;}
"has" {yylval.node = new TreeNode("has"); return has;}
"hasKey" {yylval.node = new TreeNode("has_key"); return has_key;}
"hasLabel" {yylval.node = new TreeNode("has_label"); return has_label;}
"gt" {yylval.node = new TreeNode("gt"); return gt;}
"ge" {yylval.node = new TreeNode("ge"); return ge;}
"lt" {yylval.node = new TreeNode("lt"); return lt;}
"le" {yylval.node = new TreeNode("le"); return le;}
"eq" {yylval.node = new TreeNode("eq"); return eq;}
"ne" {yylval.node = new TreeNode("ne"); return ne;}
"udf_"[a-zA-Z_]([a-zA-Z0-9_]*) {yylval.node = new TreeNode("udf", yytext); return udf;}
"mean" {yylval.node = new TreeNode("udf", "udf_mean"); return udf;}
"min" {yylval.node = new TreeNode("udf", "udf_min"); return udf;}
"max" {yylval.node = new TreeNode("udf", "udf_max"); return udf;}
[a-zA-Z_]([a-zA-Z0-9_]*) {yylval.node = new TreeNode("p", yytext); return p;}
[\+\-]?[0-9]+\.[0-9]+ {yylval.node = new TreeNode("num", yytext); return num;}
[\+\-]?[0-9]+ {yylval.node = new TreeNode("num", yytext); return num;}
"[" {yylval.node = new TreeNode("["); return l;}
"]" {yylval.node = new TreeNode("]"); return r;}
%%
