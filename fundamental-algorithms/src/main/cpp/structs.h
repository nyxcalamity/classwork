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

#include <vector>

//Provides various structures (currently limited to trees).

// Represents a node of a binary tree with int values.
class BNode {
    private:
        BNode *left;
        BNode *right;
        int height;
        int value;
    public:
        BNode(){}
        ~BNode(){}

        //Inserts a new value in the tree and rebalances the tree if necessary.
        BNode avl_insert(int value);
        //Looks for a provided value in the tree.
        BNode search(int value);
}