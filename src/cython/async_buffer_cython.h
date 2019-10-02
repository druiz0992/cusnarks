template <class T> 
class AsyncBuf {
    private:
        T *data;
        uint32_t max_nelems;

    public:
 
        T * getBuf(void);
        uint32_t  getNelems(void);
        uint32_t setBuf(T *in_data, uint32_t nelems);   
        AsyncBuf(uint32_t nelems);
        ~AsyncBuf();
};
