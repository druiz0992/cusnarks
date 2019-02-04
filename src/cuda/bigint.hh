class BigInt {
    const uint32_t BIGINT_NWORDS = 
    const uint32_t BIGINT_XOFFSET = ;
    const uint32_t BIGINT_YOFFSET = ;
    const uint32_t BIGINT_ZOFFSET = ;

    private:
        uint32_t *array_device;
        uint32_t *array_host;
        uint32_t len;

    public:
        BigInt(uint32_t *vector, uint32_t len);
        ~BigInt();
        void BigInt_ModAdd256();
        void retrieve();
}
