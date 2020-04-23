#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory["~/census-diff-privacy"]

L1 = LaplaceDistribution[0, 1/e1]
L2 = LaplaceDistribution[0, 1/e2]
L3 = LaplaceDistribution[0, 1/e3]


AbsExpectationE1j[n2_]:= 
	Expectation[Abs[x/n2 + y - Module[{k},Sum[Subscript[x, k],{k,1,n2}]]/n2], 
			   {x\[Distributed] L1, y \[Distributed] L2, 
				Reduce`FreeVariables[Module[{k},Sum[Subscript[x, k],{k,1,n2}]]]\[Distributed]ProductDistribution[{L2, n2}]}]


AbsExpectationE1jk[n2_, n3_] := 
	Expectation[Abs[u-Module[{k},Sum[Subscript[v, k],{k,1,n3}]]/n3 + (x/n2 + y - Module[{k},Sum[Subscript[x, k],{k,1,n2}]]/n2)/n3],
			   {x\[Distributed]L1, y\[Distributed]L2, Reduce`FreeVariables[Module[{k},Sum[Subscript[x, k],{k,1,n2}]]]\[Distributed]ProductDistribution[{L2, n2}],
			    u\[Distributed]L3, Reduce`FreeVariables[Module[{k},Sum[Subscript[v, k],{k,1,n3}]]] \[Distributed]ProductDistribution[{L3, n3}]}]

AbsErrorTableE1jk = ParallelTable[Print[{n2,n3}];
                                 exp = AbsExpectationE1jk[n2, n3];
                                 Save["AbsError_n2_" <> ToString[n2] <> "_n3_" <> ToString[n3] <> ".ml", exp];
                                 exp,{n2,5},{n3,5}]

Save["AbsErrorTableE1jk2.wl", AbsErrorTableE1jk]