//
//  main.cpp
//  剑指offer
//
//  Created by 蔡欣东 on 16/4/24.
//  Copyright © 2016年 蔡欣东. All rights reserved.
//

#include <iostream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>


using namespace std;

struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x):val(x), next(NULL) {
        
    }
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};

struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
    label(x), next(NULL), random(NULL) {
    }
};

class Temp {
public:
    static int N;
    
    static int SUM;
    
    Temp() {
        N++;
        SUM += N;
    }
    
};
int Temp::N = 0;
int Temp::SUM = 0;

class Solution {
    
public:
    
#pragma mark - ====================排序算法====================
    void swapFunc(int &a, int &b){
        int temp = a;
        a = b;
        b = temp;
    }
    
#pragma mark - 直接插入排序
    void insertSort(int a[],int size){
        
        for (int i = 1; i < size; i ++) {
            if (a[i] < a[i-1]) {
                int j = i - 1;
                int x = a[i];
                a[i]  = a[j];
                while (x < a[j]) {
                    a[j+1] = a[j];
                    j--;
                }
                a[j+1] = x;
            }
        }
    }
    
#pragma mark - 希尔排序
    void normalInsertSort(int a[], int d, int size){
        for (int i = d; d < size; i ++) {
            if (a[i] < a[i - d]) {
                int j = i -d;
                int x = a[i];
                a[i]  = a[j];
                while (x < a[j]) {
                    a[j+d] = a[j];
                    j = j - d;
                }
                a[j+d] = x;
            }
        }
    }
    
    
    void shellSort(int a[],int size){
        int d = size/2;
        while (d >= 1) {
            normalInsertSort(a, d, size);
            d = d/2;
        }
    }
    
#pragma mark - 选择排序
    void selectSort(int a[], int size){
        for (int i = 0; i < size; i++) {
            int k = i;
            for (int j = i+1; j<size; j++) {
                if (a[j] < a[k]) {
                    k = j;
                }
            }
            swapFunc(a[i], a[k]);
        }
    }
    
#pragma mark - 冒泡排序
    void bubbleSort(int a[], int size){
        for (int i=0; i<size; i++) {
            for (int j = 0; j<size-i-1; j++) {
                if (a[j] > a[j+1]) {
                    swapFunc(a[j], a[j+1]);
                }
            }
        }
    }
    
#pragma mark - 堆排序
    void adjustHeap(int i, int size, int a[]){
        int left = 2*(i+1)-1;
        int right = 2*(i+1);
        int max = i;
        if (i <= size/2 - 1) {
            if (left < size && a[left] > a[max]) max = left;
            if (right < size && a[right] > a[max]) max = right;
            if (max != i) swapFunc(a[max], a[i]),adjustHeap(max, size, a);
        }
    }
    
    void setupHeap(int a[], int size){
        for (int i = size/2-1; i>=0; i--) {
            adjustHeap(i, size, a);
        }
    }
    
    void heapSort(int a[], int size) {
        setupHeap(a, size);
        for (int i = size-1 ; i > 0 ; i--) {
            swapFunc(a[0], a[i]);
            adjustHeap(0, size, a);
        }
    }
    
#pragma mark - 快速排序
    int partion(int start, int end, int a[]) {
        int i = start;
        int j = end;
        int x = a[i];
        while (i < j) {
            while (i < j && a[j] > x) j--;
            if (i < j) a[i] = a[j], i++;
            while (i < j && a[i] < x) i++;
            if (i < j) a[j] = a[i], j--;
        }
        a[i] = x;
        return i;
    }
    
    void subQuickSort(int start, int end, int a[]){
        if (start < end) {
            int p = partion(start, end, a);
            subQuickSort(start, p-1 , a);
            subQuickSort(p+1 , end, a);
        }
    }
    
    void quickSort(int a[], int size){
        subQuickSort(0, size-1, a);
    }
    
#pragma mark - 归并排序
    void mergeArray(int start,int mid, int end,int a[], int tmp[]){
        int i = start;
        int n = mid;
        int j = mid+1;
        int m = end;
        int k = 0;
        while (i <=n && j<=m) {
            if (a[i] < a[j]) {
                tmp[k++] = a[i++];
            }else {
                tmp[k++] = a[j++];
            }
        }
        while (i <= n) {
            tmp[k++] = a[i++];
        }
        while (j <= m) {
            tmp[k++] = a[j++];
        }
        for (int i = 0; i < k; i++) {
            a[start+i] = tmp[i];
        }
    }
    
    void subMergeSort(int start,int end ,int a[], int tmp[]){
        if (start < end) {
            int mid = (start+end)/2;
            subMergeSort(start, mid, a, tmp);
            subMergeSort(mid+1, end, a, tmp);
            mergeArray(start, mid, end, a, tmp);
        }
    }
    
    void mergeSort(int a[],int size){
        int *tmp = new int[size];
        subMergeSort(0, size-1, a, tmp);
        delete [] tmp;
    }
    
#pragma mark - 最小 K 个数(堆排序的延伸)
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> result = {};
        
        if (input.size() < k || k < 1) {
            return result;
        }
        
        multiset<int,greater<int>> heap;
        for (size_t i = 0; i < input.size() ; i++) {
            if (heap.size() < k) {
                heap.insert(input[i]);
            }else {
                if (input[i] < *(heap.begin())) {
                    heap.erase(heap.begin());
                    heap.insert(input[i]);
                }
            }
        }
        
        for (auto pos = heap.begin(); pos != heap.end(); pos++) {
            result.push_back(*pos);
        }
        return  result;
    }
    
#pragma mark - ====================字符串====================
    
#pragma mark - 翻转单词
    
    //利用vector
    string reverseWords(string str) {
        if (str.length()<=1) {
            return str;
        }
        vector<string> ve;
        size_t i = 0;
        size_t n = str.length();
        while (i<n && str[i]==' ') {
            i++;
        }
        while (n>0 && str[n-1]==' ' ) {
            n--;
        }
        size_t index = 0;
        while (index != -1 && index < n) {
            index = str.find_first_of(" ",i);
            string newStr = str.substr(i,index-i);
            ve.push_back(newStr);
            while (index < n && str[index] == ' ') {
                index++;
            }
            i = index;
        }
        
        if (ve.empty()) {
            str = "";
        }else{
            size_t size = ve.size();
            str = "";
            for (size_t i = size -1;i>0 ; i--) {
                str = str + ve[i] + " ";
            }
            str = str + ve[0];
        }
        return str;
        
    }
    

    //先翻转每个单词，再翻转整个句子
    void reverseWord02(string &s){
        size_t i = 0;
        size_t to = 0;
        while (i < s.length()) {
            while (i < s.length() && s[i] == ' ')  i++;
            if (i == s.length()) break;
            size_t wordBegin = i;
            while (i < s.length() && s[i] != ' ') i++;
            size_t workLen = i - wordBegin;
            for (size_t j = wordBegin; j < i; j++) {
                s[to++] = s[j];
            }
            reverse(s.begin() + to - workLen, s.begin() + to);
            s[to++] = ' ';
        }
        s.resize(to > 0? to-1:to);
        reverse(s.begin(), s.end());
    }
    
#pragma mark - 左旋转字符串
    //跟翻转单词一个道理，翻转3次
    void reverseStr(string &s,int start, int end) {
        for (int i = start, j = end; i < j; i++,j--) {
            char tmp = s[i];
            s[i] = s[j];
            s[j] = tmp;
        }
    }
    
    string LeftRotateString(string str, int n) {
        if (str.length() == 0 || n <= 0) {
            return str;
        }
        int len = (int)str.length();
        reverseStr(str, 0, n-1);
        reverseStr(str, n, len-1);
        reverseStr(str, 0, len-1);
        return str;
    }
    
#pragma mark - 字符串全排列
    /**
     * 递归
     **/
    void swap(char &a,char &b){
        char tmp = a;
        a = b;
        b = tmp;
    }
    
    //判断当前交换位置的数字之前有没有出现，出现过就不交换
    bool isToSwap(string str,int start,int end){
        for (int i = start; i<end; i++) {
            if (str[i]==str[end]) {
                return false;
            }
        }
        return true;
    }
    
    //k表示当前选取到第几个数,n表示共有多少数.
    void AllRange(vector<string> &ve, string str,int k,int n){
        if (k==n) {
            ve.push_back(str);
        }
        //当前第k位的数跟它后面的数交换
        for (int i = k; i<n; i++) {
            if (isToSwap(str, k, i)) {
                swap(str[k], str[i]);
                AllRange(ve,str, k+1,n);
                swap(str[k], str[i]);
            }
        }
        
    }
    
    vector<string> Permutation(string str) {
        vector<string> ve;
        if (str.empty()) {
            return ve;
        }
        AllRange(ve, str, 0,(int)str.length());
        sort(ve.begin(), ve.end());
        return ve;
    }

#pragma mark - 最长回文字符串
    string longestPalindrome(string s) {
        if (s.length()<=1) {
            return s;
        }
        int n = (int)s.length();
        int max = 0;
        int maxLeft = 0;
        int maxRight = 0;
        for(int i=0;i<n;i++){
            //假定回文子串的长度为奇数
            int start = i-1;
            int end = i+1;
            int len = 1;
            int left = start;
            int right = end;
            while (start>=0 && end<n) {
                if (s[start]==s[end]) {
                    len = len+2;
                    left = start;
                    right = end;
                    start--;
                    end++;
                }else{
                    break;
                }
            }
            if (len>max) {
                max = len;
                maxLeft = left;
                maxRight = right;
            }
            //假定回文子串的长度为偶数
            start = i;
            end = i+1;
            len = 0;
            left = start;
            right = end;
            while (start>=0 && end<n) {
                if (s[start]==s[end]) {
                    len = len+2;
                    left = start;
                    right = end;
                    start--;
                    end++;
                }else{
                    break;
                }
            }
            if (len>max) {
                max = len;
                maxLeft = left;
                maxRight = right;
            }
        }
        return s.substr(maxLeft,maxRight-maxLeft+1);
    }
    
#pragma mark - 最长重复子串
    int lengthOfLongestSubstring(string s) {
        //保存每一个字母最近出现的位置
        int lastPos[256];
        memset(lastPos,-1,sizeof(lastPos));
        int maxLen = 0;
        int left = 0;
        int n = (int)s.length();
        for(int i = 0;i<n;i++){
            if(lastPos[s[i]]>=left){
                left = lastPos[s[i]]+1;
            }
            lastPos[s[i]] = i;
            maxLen = max(maxLen, i - left + 1);
        }
        return maxLen;
    }
    
#pragma mark - 字符串转数字
    /**
     * 前面有空格
     * 不规格输出++，--
     * 前面有数字就输出到数字就可以了，例如 123+as 输出 123
     * 判断是否超过int的范围（-2^31~2^31-1）
     **/
    int myAtoi(string str) {
        int n = (int)str.length();
        long long num = 0;//注意
        int i = 0;
        int flag = 1;
        while (str[i]==' ') {
            i++;
        }
        if (str[i]=='-') {
            flag = -1;
            i++;
        }else if (str[i]=='+'){
            flag = 1;
            i++;
        }
        for (; i<n; i++) {
            if (str[i]>='0'&&str[i]<='9') {
                num = num*10+(str[i]-'0')*flag;
            }else{
                return (int)num;
            }
            if (num>2147483647) {
                return 2147483647;
            }else if (num<-2147483648){
                return -2147483648;
            }
            
        }
        
        return (int)num;
    }
    
#pragma mark - 替换空格
    void replaceSpace(char *str,int length) {
        int spaceCount = 0;
        for (int i=0; i<length; i++) {
            if (str[i]==' ') {
                spaceCount++;
            }
        }
        if (spaceCount==0) {
            return;
        }
        
        int newLen = length+2*spaceCount;
        char* index = str + length;
        while (index>=str) {
            if (*index==' ') {
                str[newLen--] = '0';
                str[newLen--] = '2';
                str[newLen--] = '%';
            }else{
                str[newLen--] = *index;
            }
            index--;
        }
    }
    
#pragma mark - ====================数组====================
    
#pragma mark - 二维数组中的查找
    bool Find(vector<vector<int> > array,int target) {
        int n = 0;
        int m = (int)array[0].size()-1;
        while(n<array.size()&&m>=0){
            if(target>array[n][m]){
                n++;
            }else if(target<array[n][m]){
                m--;
            }else{
                return true;
            }
            
        }
        return false;
    }
    
#pragma mark - 旋转数组的最小数字
    int minNumberInRotateArray(vector<int> rotateArray) {
        if (rotateArray.size()==0) {
            return 0;
        }else{
            int first = 0;
            int last = (int)rotateArray.size()-1;
            int mid  = 0;
            while (rotateArray[first]>=rotateArray[last]) {
                //终止条件
                if (last-first==1) {
                    return rotateArray[last];
                }
                mid = (first+last)/2;
                //如果中间的数比first的大说明最小值到后半段，否则在前半段
                if (rotateArray[mid]>=rotateArray[first]) {
                    first = mid;
                }else{
                    last = mid;
                }
                //无法判断大小的时候，只能顺序查找，如{1,0,1,1,1}
                if (rotateArray[first]==rotateArray[mid]&&rotateArray[last]==rotateArray[mid]) {
                    int min = rotateArray[first];
                    for (int i = first+1; i<=last; i++) {
                        if (rotateArray[i]<min) {
                            min = rotateArray[i];
                        }
                    }
                    return min;
                }
               
            }
            return rotateArray[first];
        }
    }

#pragma mark - 二分查找
    
    int binary_search(int start, int end, int key, int a[]) {
        int mid = 0;
        while (start <= end) {
            mid = (start + end)/2;
            if (a[mid] > key) {
                end = mid - 1;
            }else if (a[mid] < key){
                start = mid + 1;
            }else {
                return mid;
            }
        }
        return -1;
    }
    
#pragma mark - 数组中出现次数超过一半的数字
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        unordered_map<int, int> map;
        int n = (int)numbers.size();
        if (n==1) {
            return numbers[0];
        }
        for (int i = 0; i<n; i++) {
            if (map.count(numbers[i])>0) {
                int count = map[numbers[i]];
                count++;
                map[numbers[i]] = count;
                if (count>n/2) {
                    return numbers[i];
                }
            }else{
                map[numbers[i]] = 1;
            }
        }
        
        
        return 0;
    }
    
    
    int MoreThanHalfNum_Solution02(vector<int> numbers) {
        int n = (int)numbers.size();
    
        if (n == 0) {
            return 0;
        }
        
        int result = numbers[0];
        int time = 1;
        for (int i = 1; i < n; i++) {
            if (time == 0) {
                result = numbers[i];
                time = 1;
            }else if (result == numbers[i]) {
                time++;
            }else {
                time--;
            }
        }
        
        int count = 1;
        for (int i = 0; i < n; i++) {
            if (result == numbers[i]) {
                count++;
            }
        }
        if (count > n/2) {
            return result;
        }
        return 0;
    }
    
#pragma mark - 第 k 大的数
    /**
     * 堆排序（用STL的set也可以）
     **/
    void HeapAdjust(vector<int>& nums,int i, int size) {
        int left = 2*(i+1)-1;
        int right = 2*(i+1);
        int max = i;
        if (i <= size/2-1) {
            if (left < size && nums[left] > nums[max]) {
                max = left;
            }
            if (right < size && nums[right] > nums[max]) {
                max = right;
            }
            if (max != i) {
                swapFunc(nums[i], nums[max]);
                HeapAdjust(nums, max,size);
            }
        }
    }
    
    int findKthLargest(vector<int>& nums, int k) {
        int n = (int)nums.size();
        if (k > n) {
            return -1;
        }
        for (int i = n/2-1; i >= 0; i--) {
            HeapAdjust(nums, i, n);
        }
        for (int i = 1; i < k; i++) {
            swapFunc(nums[0], nums[n-i]);
            HeapAdjust(nums, 0 ,n-i);
        }
        return nums[0];
    }
    
#pragma mark - 把数组排成最小的数
    /**排序规则如下：
    * 若ab > ba 则 a > b，
    * 若ab < ba 则 a < b，
    * 若ab = ba 则 a = b；
    * 解释说明：
    * 比如 "3" < "31"但是 "331" > "313"，所以要将二者拼接起来进行比较
    */
    static bool compare(const string &s1 ,const string &s2){
        string ss1 = s1 + s2;
        string ss2 = s2 + s1;
        return ss1 < ss2;
    }
    
    /**
     *  先将整型数组转换成String数组，然后将String数组排序，最后将排好序的字符串数组拼接出来。关键就是制定排序规则
     *
     */
    string PrintMinNumber(vector<int> numbers) {
        if (numbers.size() <= 0) {
            return "";
        }
        
        vector<string> result;
        
        for (size_t i = 0; i < numbers.size(); i++) {
            stringstream ss;
            ss<<numbers[i];
            result.push_back(ss.str());
        }
        
        sort(result.begin(), result.end(), compare);
        
        string resultStr = "";
        
        for (size_t i = 0; i < result.size(); i++) {
            resultStr.append(result[i]);
        }
        
        return resultStr;
    }
    
#pragma mark - 数组的逆序对
    //采用归并解决
    int mergeSort(vector<int> &data, int start, int end,vector<int> &tmp){
        if (start == end) {
            return 0;
        }
        int mid = (start + end)/2;
        
        int leftCount = mergeSort(data, start, mid, tmp)%1000000007 ;
        
        int rightCount = mergeSort(data, mid+1, end, tmp)%1000000007 ;
    
        
        int leltBack = mid;
        
        int rightBack = end;
        
        int count = 0;
        
        int tmpIndex = end;
        
        while (start <= leltBack && mid+1 <= rightBack) {
            if (data[leltBack] > data[rightBack]) {
                tmp[tmpIndex--] = data[leltBack--];
                count += rightBack - mid;
                if (count > 1000000007 ) {
                    count %= 1000000007 ;
                }
            }else {
                tmp[tmpIndex--] = data[rightBack--];
            }
        }
        
        while (start <= leltBack) {
            tmp[tmpIndex--] = data[leltBack--];
        }
        
        while (mid+1 <= rightBack) {
            tmp[tmpIndex--] = data[rightBack--];
        }
        
        for (int i = start; i <= end; i++) {
            data[i] = tmp[i];
        }
        
        return (count + leftCount + rightCount)%1000000007 ;
        
    }
    
    int InversePairs(vector<int> data) {
        if ( data.size() == 0) {
            return 0;
        }
        int len = (int)data.size();
        
        vector<int> tmp(data);
        
        return mergeSort(data, 0, len-1, tmp);
    }
    
#pragma mark - 数字在排序数组中出现的次数
    /**
     *  看到排序要想到二分查找，通过二分查找找到k第一次出现的位置和最后一次出现的位置，就可以计算出次数
     *
     */
    
    int findFirstK(vector<int> data, int start, int end, int k) {
        int mid = -1;
        while (start <= end) {
            mid = (start + end)/2;
            if (data[mid] > k) {
                end = mid - 1;
            }else if (data[mid] < k) {
                start = mid + 1;
            }else {
                if (mid == 0 || (mid  > 0 && data[mid-1] != k)) {
                    return mid;
                }else {
                    end = mid - 1;
                }
            }
        }
        return -1;
    }
    
    int findLastK(vector<int> data, int start, int end, int k){
        int mid = -1;
        int len = (int)data.size();
        while (start <= end) {
            mid = (start + end)/2;
            if (data[mid] > k) {
                end = mid - 1;
            }else if (data[mid] < k) {
                start = mid + 1;
            }else {
                if (mid == len-1 || (mid  < len-1 && data[mid+1] != k)) {
                    return mid;
                }else {
                    start = mid + 1;
                }
            }
        }
        return -1;
    }
    
    int GetNumberOfK(vector<int> data ,int k) {
        if (data.size() == 0) {
            return 0;
        }
        int size = (int)data.size();
        int firstIndex = findFirstK(data, 0, size-1, k);
        int lastIndex = findLastK(data, 0, size-1, k);
        if (firstIndex > -1 && lastIndex > -1) {
            return lastIndex - firstIndex + 1;
        }
        return 0;
        
    }
    
#pragma mark - ====================链表====================
    
#pragma mark - 翻转链表
    ListNode* reverseList(ListNode* head) {
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *pre = head;
        ListNode *p = head->next;
        ListNode *next = NULL;
        while (p != NULL) {
            next = p->next;
            p->next = pre;
            pre = p;
            p = next;
        }
        head->next = NULL;
        return pre;
    }
    
#pragma mark - 删除某个结点
    void deleteNode(ListNode* node) {
        //暂时包含后面那个结点
        ListNode *tmp = node->next;
        //node结点赋值下个结点的值以及next指针
        node->val = node->next->val;
        //删除后面那个结点
        node->next = node->next->next;
        delete tmp;
    }

#pragma mark - 删除倒数第n个结点
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        if (head == NULL) {
            return head;
        }
        ListNode *newHead = new ListNode(0);
        newHead->next = head;
        ListNode *pre = newHead;
        ListNode *p = newHead;
        for (int i = 0; i < n; i++) {
            p = p->next;
        }
        while (p->next != NULL) {
            pre = pre->next;
            p = p->next;
        }
        pre->next = pre->next->next;
        return newHead->next;
    }
    
#pragma mark - 合并两个有序的链表
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == NULL) {
            return l2;
        }
        if (l2 == NULL) {
            return l1;
        }
        ListNode *newHead = new ListNode(0);
        ListNode *p = newHead;
        ListNode *p1 = l1;
        ListNode *p2 = l2;
        while (p1 != NULL && p2 != NULL ) {
            if (p1->val < p2->val) {
                p->next = p1;
                p1 = p1->next;
            }else {
                p->next = p2;
                p2 = p2->next;
            }
  
            p  = p ->next;
        }
        while (p1 != NULL) {
            p->next = p1;
            p1 = p1->next;
            p  = p->next;
        }
        while (p2 != NULL) {
            p->next = p2;
            p2 = p2->next;
            p  = p->next;
        }
        p->next = NULL;
        return newHead->next;
    }
    
#pragma mark - 判断链表是否有环
    bool hasCycle(ListNode *head) {
        if (head==NULL || head->next == NULL) {
            return false;
        }
        //定义一个快指针和慢指针
        ListNode *slow = head;
        ListNode *fast = head;
        while (fast != NULL && fast->next != NULL) {
            slow = slow->next;
            fast = fast->next->next;
            //注意要判断fast不能为NULL，当快指针赶上慢指针的时候说明有环
            if (fast != NULL && (fast == slow || fast->next == slow)) {
                return true;
            }
        }
        return false;
    }
    
#pragma mark - 两个链表的第一个公共节点
    /**
     * 第一种情况 两个链表都没有环，那么他们是Y形相交，计算他们的长度，求出长度差k，分别给两个链表定义两个遍历的指针，长链表的指针先走k步，之后两个链表的指针同时走，一直到相同的结点；
     * 第二种情况，两个链表都有环，有两种交点，一种是进入环前有交点，另一种是链表入环的交点
     * 不可能一个有环一个无环，一个单链表最后一个结点的next是指向NULL，而有环的链表的结点的next不指向NULL，所以不可能有结点相交
     
     **/
    ListNode* FindFirstCommonNode( ListNode *pHead1, ListNode *pHead2) {
        int len1 = getListLen(pHead1);
        int len2 = getListLen(pHead2);
        int diff = len1 - len2;
        ListNode *longP = pHead1;
        ListNode *shortP = pHead2;
        if (len2 > len1) {
            longP = pHead2;
            shortP = pHead1;
            diff = len2 - len1;
        }
        for (int i = 0; i < diff; i++) {
            longP = longP->next;
        }
        
        while (longP != NULL && shortP != NULL && longP != shortP) {
            longP = longP->next;
            shortP = shortP->next;
        }
        
        return shortP;
        
    }
    
    int getListLen(ListNode *head){
  
        int count = 0;
        ListNode *p = head;
        while (p != NULL) {
            p = p->next;
            count++;
        }
        return count;
    }
    
#pragma mark - 删除链表中重复节点
    ListNode* deleteDuplicates(ListNode* head){
        if (head == NULL || head->next == NULL) {
            return head;
        }
        ListNode *pre = head;
        ListNode *p = pre->next;
        while (p != NULL) {
            if (pre->val == p->val) {
                pre->next = p->next;
                p = p->next;
            }else {
                pre = pre->next;
                p = p->next;
            }
        }
        return head;
    }
    
    
#pragma mark - 复杂链表的复制
    
    /**
     复制所有点

     @param head
     */
    void cloneNodes(RandomListNode * head){
        RandomListNode *node = head;
        while (node != NULL) {
            RandomListNode *cloneNode = new RandomListNode(node->label);
            cloneNode->next = node->next;
            
            node->next = cloneNode;
            node = cloneNode->next;
        }
    }
    
    
    /**
     设置random结点，复制的结点的random结点是原本结点的random结点的下一个

     @param head
     */
    void cloneRandomNodes(RandomListNode *head){
        RandomListNode *node = head;
        while (node != NULL) {
            RandomListNode *cloneNodel = node->next;
            if (node->random != NULL) {
                cloneNodel->random = node->random->next;
            }
            node = cloneNodel->next;
        }
    }
    
    
    /**
     拆链表

     @param head

     @return
     */
    RandomListNode* reconnectNodes(RandomListNode *head){
        RandomListNode *node = head;
        RandomListNode *cloneHead = NULL;
        RandomListNode *cloneNode = NULL;
        
        if (node != NULL) {
            cloneHead = node->next;
            cloneNode = cloneHead;
            node->next = cloneNode->next;
            node = node->next;
        }
        
        while (node != NULL) {
            cloneNode->next = node->next;
            cloneNode = cloneNode->next;
            node->next = cloneNode->next;
            node = node->next;
        }
        return cloneHead;
    }
    
    
    /**
     合并

     @param pHead

     @return
     */
    RandomListNode* Clone(RandomListNode* pHead) {
        cloneNodes(pHead);
        cloneRandomNodes(pHead);
        return  reconnectNodes(pHead);
    }
    
#pragma mark - 二叉搜索树转化为有序双链表
    //遍历到的结点左边接左子树最大的结点，右边接右子树最小的结点
    TreeNode* Convert(TreeNode* pRootOfTree) {
        if (pRootOfTree == NULL || (pRootOfTree->left==NULL && pRootOfTree->right==NULL)) {
            return pRootOfTree;
        }
        
        TreeNode *leftHead = Convert(pRootOfTree->left);
        TreeNode *pLeft = leftHead;
        //获取左边链表的最后一个结点
        while (pLeft!=NULL && pLeft->right != NULL) {
            pLeft = pLeft->right;
        }
        
        if (pLeft != NULL) {
            pLeft->right = pRootOfTree;
            pRootOfTree->left = pLeft;
        }
        
        //获取右边链表的第一个结点
        TreeNode *rightHead = Convert(pRootOfTree->right);
        if (rightHead != NULL) {
            rightHead->left = pRootOfTree;
            pRootOfTree->right = rightHead;
        }
        
        return leftHead!=NULL? leftHead:pRootOfTree;
    }
    
#pragma mark - ====================二叉树====================
    
#pragma mark - 根据中序和前序遍历结果重建二叉树
    /**
     *
     *
     *  @param pre      前序遍历
     *  @param in       中序遍历
     *  @param preStart 前序开始位置
     *  @param inStart  中序开始位置
     *  @param length   子序列的长度
     *
     *  @return
     */
    TreeNode* subReConstructBinaryTree(vector<int> &pre,vector<int> &in,int preStart,int inStart,int length) {
        if (length <= 0) {
            return NULL;
        }
        TreeNode *root = new TreeNode(pre[preStart]);
        int len = 0;
        int i = 0;
        //一直找到中序遍历中与前序遍历第一个数相等的，获取子序列长度
        while (i < inStart+length && in[i] != pre[preStart]) {
            len = (++i)-inStart;
        }
        root->left = subReConstructBinaryTree(pre, in, preStart+1, inStart, len);
        root->right = subReConstructBinaryTree(pre, in, preStart + len+ 1, inStart + len + 1, length-len-1);
        return root;
        
    }
    
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return subReConstructBinaryTree(preorder, inorder, 0, 0, (int)preorder.size());
    }
    
#pragma mark - 根据中序和后序遍历结果重建二叉树
    
    TreeNode* subReConstructBinaryTree02(vector<int> &post,vector<int> &in,int postStart,int postEnd,int inStart,int inEnd) {
        if (postStart > postEnd) {
            return NULL;
        }
        TreeNode *root = new TreeNode(post[postEnd]);
        int p = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (in[i] == post[postEnd]) {
                p = i;
                break;
            }
        }
        int len = p - inStart;
        //参考上题
        root->left = subReConstructBinaryTree02(post, in, postStart, postStart+len-1, inStart, p-1);
        root->right = subReConstructBinaryTree02(post, in, postStart+len, postEnd-1, p+1, inEnd);

        return root;
        
    }
    
    TreeNode* buildTree02(vector<int>& inorder, vector<int>& postorder) {
        return subReConstructBinaryTree02(postorder, inorder, 0, (int)postorder.size()-1, 0, (int)inorder.size()-1);
    }
    
#pragma mark - 翻转二叉树
    TreeNode* invertTree(TreeNode* root) {
        if (root) {
            invertTree(root->left);
            invertTree(root->right);
            //翻转
            TreeNode *tmp = root->left;
            root->left = root->right;
            root->right = tmp;
        }
        return root;
    }
    
#pragma mark - 从上往下打印二叉树
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ve;
        if (root == NULL) {
            return ve;
        }
        queue<TreeNode *> queue;
        queue.push(root);
        queue.push(NULL);
        vector<int> current_ve;
        while (!queue.empty()) {
            TreeNode *node = queue.front();
            queue.pop();
            if (node == NULL) {
                ve.push_back(current_ve);
                current_ve.resize(0);
                //关键
                if (queue.size()>0) {
                    queue.push(NULL);
                }
            }else {
                current_ve.push_back(node->val);
                if (node->left != NULL) {
                    queue.push(node->left);
                }
                if (node->right != NULL) {
                    queue.push(node->right);
                }
            }
        }
        return ve;
    }
    
#pragma mark - 判断某个数组是不是二叉树的后序遍历结果
    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.size() <= 0) {
            return false;
        }
        size_t i = 0;
        size_t size = sequence.size();
        while (--size > 0) {
            //关键下面两步，左子树一定比根小，右子树一定比根大
            while(sequence[i++] < sequence[size]);
            while(sequence[i++] > sequence[size]);
            if (i < size) {
                return false;
            }
            i = 0;
        }
        return true;
    }
#pragma mark - 判断二叉树中是否有和为某个值的路径
    
    bool hasPathSum(TreeNode* root, int sum) {
        if (root == NULL) {
            return false;
        }
        if (root->left == NULL && root->right == NULL && root->val != sum) {
            return false;
        }
        return testHasPathSum(root, sum);
    }
    
    /**
     *  利用树的深度遍历
     *
     *  @param node
     *  @param sum
     *
     *  @return
     */
    bool testHasPathSum(TreeNode *node, int sum) {
        if (node->left == NULL && node->right == NULL && node->val == sum) {
            return true;
        }
        if (node->left == NULL && node->right == NULL && node->val != sum) {
            return false;
        }
        bool testLeft = false;
        bool testRight = false;
        if (node->left != NULL) {
            testLeft = testHasPathSum(node->left, sum-node->val);
        }
        if (node->right != NULL) {
            testRight = testHasPathSum(node->right, sum-node->val);
        }
        return testLeft|testRight;
    }
    
#pragma mark - 二叉树中和为某个值的路径
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int>> result;
        if (root == NULL) {
            return result;
        }
        if (root->left == NULL && root->right == NULL && root->val != sum) {
            return result;
        }
        vector<int> tmp;
        dfs(root, sum, tmp, result);
        return result;
    }
    
    /**
     *  利用树的深度遍历
     *
     *  @param node
     *  @param sum
     *  @param tmpResult
     *  @param result
     */
    void dfs(TreeNode *node,int sum, vector<int> tmpResult, vector<vector<int>> & result){
        tmpResult.push_back(node->val);
        if (node->left == NULL && node->right == NULL && node->val == sum) {
            result.push_back(tmpResult);
            return;
        }
        if (node->left == NULL && node->right == NULL && node->val != sum) {
            return;
        }
        if (node->left != NULL) {
            dfs(node->left, sum-node->val, tmpResult, result);
        }
        if (node->right != NULL) {
            dfs(node->right, sum-node->val, tmpResult, result);
        }
    }

#pragma mark - 二叉树的镜像
    void Mirror(TreeNode *pRoot) {
        if (pRoot==NULL) {
            return;
        }else{
            TreeNode* tmp = pRoot->left;
            pRoot->left = pRoot->right;
            pRoot->right = tmp;
            Mirror(pRoot->left);
            Mirror(pRoot->right);
        }
    }
    
#pragma mark - 二叉树的下一个结点
    TreeLinkNode* GetNext(TreeLinkNode* pNode) {
        if (pNode == NULL) {
            return NULL;
        }
        if (pNode->right != NULL) {
            TreeLinkNode *p = pNode->right;
            //找到右子树最左边的结点
            while (p->left !=NULL) {
                p = p->left;
            }
            return p;
        }
        //当pNode不是根结点的时候
        while (pNode->next != NULL) {
            if (pNode->next->left == pNode) {
                return pNode->next;
            }
            pNode = pNode->next;
        }
        return NULL;
    }
    
#pragma mark - 输入两颗二叉树A，B，判断B是不是A的子结构。
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2){
        if (pRoot1==NULL||pRoot2==NULL) {
            return false;
        }
        return isSubTree(pRoot1, pRoot2)||HasSubtree(pRoot1->left, pRoot2)||HasSubtree(pRoot1->right, pRoot2);
        
    }
    
    bool isSubTree(TreeNode* pRoot1,TreeNode* pRoot2){
        if (pRoot2 == NULL) return true;
        if (pRoot1 == NULL) return false;
        if (pRoot1->val==pRoot2->val) {
            return isSubTree(pRoot1->left, pRoot2->left)&&isSubTree(pRoot1->right, pRoot2->right);
        }else{
            return false;
        }
    }
    
#pragma mark - 从上往下打印二叉树
    vector<int> PrintFromTopToBottom(TreeNode *root) {
        if (root == NULL) {
            return {};
        }
        vector<int> value = {};
        queue<TreeNode *> queue;
        queue.push(root);
        while (!queue.empty()) {
            TreeNode *node = queue.front();
            value.push_back(node->val);
            queue.pop();
            if (node->left != NULL) {
                queue.push(node->left);
            }
            if (node->right != NULL) {
                queue.push(node->right);
            }
        }
        return value;
    }
    
#pragma mark - 二叉树深度
    //深度=max（左子树深度，右子树）+ 1
    int TreeDepth(TreeNode* pRoot) {
        if (pRoot == NULL) {
            return 0;
        }
        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        return left > right ? (left + 1):(right + 1);
    }
    
#pragma mark - 判断平衡二叉树
    //后序遍历，判断子树是否平衡，一直到根部
    bool IsBalanced(TreeNode* root, int *depth) {
        if (root == NULL) {
            *depth = 0;
            return true;
        }
        int left;
        int right;
        if (IsBalanced(root->left, &left) && IsBalanced(root->right, &right)) {
            int diff = abs(left - right);
            if (diff <= 1) {
                *depth = left > right ?(left + 1):(right + 1);
                return true;
            }
        }
        return false;
    }
    
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if (pRoot == NULL) {
            return true;
        }
        int depth = 0;
        return IsBalanced(pRoot, &depth);
    }
    
#pragma mark - ====================栈与队列====================
    
#pragma mark - 用两个栈实现队列
    class Queue {
    public:
        int size = 0;
        stack<int> s1;
        stack<int> s2;
        
        void push(int x) {
            s1.push(x);
            size++;
        }
        
        void pop(void) {
            if (!s2.empty()) {
                s2.pop();
            }else{
                while (!s1.empty()) {
                    int t = s1.top();
                    s1.pop();
                    s2.push(t);
                }
                s2.pop();
            }
            size--;
        }
        
        int peek(void) {
            if (!s2.empty()) {
                return s2.top();
            }else{
                while (!s1.empty()) {
                    int t = s1.top();
                    s1.pop();
                    s2.push(t);
                }
                return s2.top();
            }
        }
        
        bool empty(void) {
            if (size==0) {
                return true;
            }else{
                return false;
            }
        }
    };
    
#pragma mark - 用两个队列实现栈
    
    class Stack {
        int size=0;
        queue<int> q1;
        queue<int> q2;
    public:

        void push(int x) {
            //哪个队列不为空就往哪个插入值
            if (empty()||!q1.empty()) {
                q1.push(x);
            }else{
                q2.push(x);
            }
            size++;
        }
        
        void pop() {
            if (!q1.empty()) {
                while (q1.size()>1) {
                    int a = q1.front();
                    q1.pop();
                    q2.push(a);
                }
                q1.pop();
            }else{
                while (q2.size()>1) {
                    int a = q2.front();
                    q2.pop();
                    q1.push(a);
                }
                q2.pop();
            }
            size--;
        }

        int top() {
            if (!q1.empty()) {
                return q1.back();
            }else{
                return q2.back();
            }
        }
        
        bool empty() {
            if (size==0) {
                return true;
            }else{
                return false;
            }
        }
    };
    
#pragma mark - 实现一个栈，可以用常数级时间找出栈中的最小值
    
    class MinStack {
        stack<int> s;
        stack<int> minStack;
    public:
        void push(int x) {
            if (s.empty()) {
                s.push(x);
                minStack.push(x);
            }else{
                s.push(x);
                int k = minStack.top();
                if (k<=x) {
                    minStack.push(k);
                }else{
                    minStack.push(x);
                }
            }
        }
        
        void pop() {
            s.pop();
            minStack.pop();
        }
        
        int top() {
            return s.top();
        }
        
        int getMin() {
            return minStack.top();
        }
    };
    
#pragma mark - 判断栈的压栈、弹栈序列是否合法
    
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if (pushV.size() == 0) {
            return false;
        }else{
            stack<int> stack;
            size_t size = pushV.size();
            size_t k = 0;
            for (size_t i = 0; i < size; i++) {
                if (!stack.empty() && stack.top() == popV[i]) {
                    stack.pop();
                }else{
                    while (true) {
                        if (k >= size) {
                            return false;
                        }else{
                            if (pushV[k] == popV[i]) {
                                k ++;
                                break;
                            }else{
                                stack.push(pushV[k]);
                                k ++;
                            }
                        }
                    }
                }
            }
            return true;
            
        }
    }
    
#pragma mark - 从尾到头打印链表
    vector<int> printListFromTailToHead(struct ListNode* head) {
        if (head==NULL) {
            return {};
        }else{
            stack<int> stack;
            vector<int> ve;
            ListNode* p = head;
            while (p!=NULL) {
                stack.push(p->val);
                p = p->next;
            }
            while (!stack.empty()) {
                int tmp = stack.top();
                ve.push_back(tmp);
                stack.pop();
            }
            return ve;
        }
    }
    
#pragma mark - 位运算
    //通过异或可以实现两个数互换
    //某个数异或-1，相当于取反
#pragma mark - 二进制有多少个1
    int hammingWeight(uint32_t n) {
        int count = 0;
        while (n != 0) {
            n = n & (n-1);
            count++;
        }
        return count;
    }
    
#pragma mark - 给一个数组，所有数字都出现了偶数次，只有一个出现了一次，找出这个数
    int singleNumber(vector<int>& nums) {
        if (nums.size() == 1) {
            return nums[0];
        }
        for (size_t i = 1; i < nums.size(); i++) {
            nums[0] ^= nums[i];
        }
        return nums[0];
    }
    
#pragma mark - 给一个数组，所有数字都出现了三次，只有一个出现了一次，找出这个数
    int singleNumber02(vector<int>& nums) {
        //去掉那个特殊的数，所有数上1的个数和是3的倍数，通过统计每一位上面1的个数，如果除以3有余数说明那个特殊的数在这个位上为1
        int bit[32] = {0};
        int result = 0;
        for (size_t i = 0; i < 32; i++) {
            for (size_t j = 0; j < nums.size(); j++) {
                bit[i] += (nums[j]>>i) & 1;
            }
            result |= (bit[i] % 3)<<i;
        }
        return result;
    }
    
#pragma mark - 给一个数组，所有数组都出现了偶数次，只有两个数字出现了一次，找出这两个数
    
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int tmpResult = 0;
        
        for (size_t i = 0; i < data.size(); i++) {
            tmpResult ^= data[i];
        }
        //获取异或结果最右边的1的位置
        int index = tmpResult - (tmpResult & (tmpResult - 1));
        
        for (size_t i = 0; i < data.size(); i ++) {
            if ((data[i]>>(index-1))%2 == 0) {
                *num1 ^= data[i];
            }else {
                *num2 ^= data[i];
            }
        }
        
    }
    
#pragma mark - 不用加减乘除做加法
    /**
     *  每一位先异或，再通过想与和左移获得进位，将两个结果异或就是结果
     *
     */
    int Add(int num1, int num2) {
        int sum;
        int carry;
        do {
            sum = num1 ^ num2;
            carry = (num1 & num2)<<1;
            num1 = sum;
            num2 = carry;
        } while (carry != 0);
        return sum;
    }
    
    
#pragma mark - ====================未分类====================
    
#pragma mark - 变态跳台阶问题
    int jumpFloorII(int number) {
        if (number<=0) {
            return -1;
        }else if (number==1){
            return 1;
        }else{
            return 1<<(number-1);
        }
    }
    
#pragma mark - 调整数组顺序使奇数位于偶数前面,利用直接插入排序的思想
     void reOrderArray(vector<int> &array) {
        for (int i=1; i<array.size(); i++) {
            if (array[i]%2>0) {
                int j = i-1;
                int x = array[i];
                while (j>=0&&array[j]%2==0) {
                    array[j+1] = array[j];
                    j--;
                }
                array[j+1] = x;
            }
        }
    }
    
#pragma mark - 顺时针打印矩阵
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> result;
        int row = (int)matrix.size();
        int col = (int)matrix[0].size();
        if (row==0||col==0) {
            return result;
        }
        int top = 0;
        int bottom = row-1;
        int left = 0;
        int right = col-1;
        while (top<=bottom&&left<=right) {
            for (int i = left; i<= right; i++) {
                result.push_back(matrix[top][i]);
            }
            for (int i = top+1; i<=bottom; i++) {
                result.push_back(matrix[i][right]);
            }
            if (top!=bottom) {
                for (int i = right-1; i>=left; i--) {
                    result.push_back(matrix[bottom][i]);
                }
            }
            if (left!=right) {
                for (int i = bottom-1; i>=top+1; i--) {
                    result.push_back(matrix[i][left]);
                }
            }
            top++;
            bottom--;
            left++;
            right--;
        }
        return  result;
    }

#pragma mark - 连续子数组的最大和
    int FindGreatestSumOfSubArray(vector<int> array) {
        if (array.empty()) {
            return 0;
        }
        int mx = INT_MIN;
        int sum = 0;
        for (int i = 0; i<(int)array.size(); i++) {
            sum = sum + array[i];
            mx = max(mx, sum);
            if (sum<0) {
                sum = 0;
            }
        }
        return mx;
    }
    
    int FindGreatestSumOfSubArray02(vector<int> array) {
        if (array.empty()) {
            return 0;
        }
        int mx = INT_MIN;
        //dp[i]指定以第i个数结尾连续子数组的最大和
        vector<int> dp;
        dp.resize(array.size(), INT_MIN);
        for (size_t i = 0; i < dp.size(); i++) {
            if (i == 0 || dp[i - 1] <= 0) {
                dp[i] = array[i];
            }else {
                dp[i] = dp[i-1] + array[i];
            }
            if (dp[i] > mx) {
                mx = dp[i];
            }
        }
        
        return mx;
    }

#pragma mark - 大数相乘
    void reverseString(string &str){
        size_t i = 0;
        size_t j = str.length()-1;
        for (; i < j; i++,j--) {
            char tmp = str[i];
            str[i] = str[j];
            str[j] = tmp;
        }
    }
    
    string multiLargeNum(string a, string b){
        size_t n = a.length();
        size_t m = b.length();
        string result(n + m - 1, '0');
        //逆序更符合我们的运算习惯
        reverseString(a);
        reverseString(b);
        //乘法进位
        int multFlag;
        //加法进位
        int addFlag;
        for (size_t i = 0; i < n ; i++) {
            multFlag = 0;
            addFlag = 0;
            for (size_t j = 0; j < m; j++) {
                int temp1 = (a[i]-'0')*(b[j]-'0')+multFlag;
                multFlag = temp1/10;
                temp1 = temp1%10;
                int temp2 = (result[i+j]-'0')+temp1+addFlag;
                addFlag = temp2/10;
                result[i+j] = '0'+temp2%10;
            }
            result[i+m] += multFlag + addFlag;
        }
        reverseString(result);
        return result;
    }
  
    
#pragma mark - 第n个丑数
    /**
     *  用动态规划,丑数的定义为只包含因子2、3和5的数称作丑数，1是最小的丑数，因此下一个丑数是由上一个丑数乘以2，3，5中的一个得到的
     *
     */
    int GetUglyNumber_Solution(int index) {
        if (index < 7) {
            return index;
        }
        vector<int> ve(index);
        
        ve[0] = 1;
        
        int t2 = 0,t3 = 0,t5 = 0;
        
        for (int i = 1; i < index; i++) {
            ve[i] = min(ve[t2]*2, min(ve[t3]*3, ve[t5]*5));
            
            if (ve[i] >= ve[t2]*2) t2++;
            
            if (ve[i] >= ve[t3]*3) t3++;
            
            if (ve[i] >= ve[t5]*5) t5++;
        }
        
        return ve[index-1];
    }
    
#pragma mark - 第一个只出现一次的字符
    //用hash表
    int FirstNotRepeatingChar(string str) {
        if (str.length() <= 0) {
            return -1;
        }
        
        unordered_map<char, int> map;
        
        for (size_t i = 0; i < str.length(); i++) {
            if (map.count(str[i]) > 0) {
                int count = map[str[i]];
                count++;
                map[str[i]] = count;
            }else {
                map[str[i]] = 1;
            }
        }
        
        int index = INT_MAX;
        
        for (auto pos = map.begin(); pos != map.end(); pos++) {
            if (pos->second == 1) {
                int tmpIndex = (int)str.find_first_of(pos->first);
                if (tmpIndex < index) {
                    index = tmpIndex;
                }
            }
        }
        
        return index;
    }
    
#pragma mark - 和为S的两个数字
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        vector<int> result;
        result.resize(0);
        if (array.size() == 0) {
            return result;
        }
        size_t n = array.size();
        size_t start = 0;
        size_t end = n-1;
        while (start <= end) {
            int tmp = array[start] + array[end];
            if (tmp > sum) {
                end--;
            }else if (tmp < sum) {
                start++;
            }else {
                result.push_back(array[start]);
                result.push_back(array[end]);
                break;
            }
        }
        
        return result;
    }
    
#pragma mark - 和为S的连续正数序列
    /**
     *  因为每个序列至少包含2个数字，序列头部从1开始，尾部从2开始,如果序列和小于sum，则尾部往后移1位再包含一个数，如果大于sum，则去掉序列头部，头部往后移一位
     *
     */
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> result;
        result.resize(0);
        int mid = (sum + 1)>>1;
        int small = 1;
        int big = 2;
        int tmpSum = small + big;
        while (small < mid) {
            
            if (tmpSum == sum) {
                result.push_back(getList(small, big));
            }
            
            while (small < mid && tmpSum > sum) {
                tmpSum -= small;
                small++;
                if (tmpSum == sum) {
                    result.push_back(getList(small, big));
                }
            }
            
            big++;
            tmpSum += big;
        }
        
        return result;
    }
    
    vector<int> getList(int start, int end){
        vector<int> list;
        list.resize(0);
        for (int i = start; i<= end; i++) {
            list.push_back(i);
        }
        return list;
    }
    
#pragma mark - 求1+2+3+...+n
    //利用构造函数
    int Sum_Solution(int n) {
        Temp *t = new Temp[n];
        delete [] t;
        t = NULL;
        return Temp::SUM;
    }
    
    //利用短路原则
    int Sum_Solution02(int n) {
        int sum = 0;
        n == 0 || (sum = Sum_Solution02(n-1));
        return sum + n;
    }
};

int main(int argc, const char * argv[]) {
 
    return 0;
}
