/*
   Copyright 2013 Denys Sobchyshak

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "structs.h"

//Inserts a new value in the tree and rebalances the tree if necessary.
BNode BNode::avl_insert(int value){
    return NULL;
}

//Looks for a provided value in the tree.
BNode BNode::search(int value){
    BNode *p = &this;

    while (p) {
        if (p->value == value) return p;
        if (p->value > value)
            p = p->left;
        else
            p = p->right;
    }

    return NULL;
}