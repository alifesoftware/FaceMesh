package com.alifesoftware.facemesh.data

import androidx.room.TypeConverter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Serializes 128-d face embeddings to compact little-endian byte blobs in Room. */
class FloatArrayConverter {

    @TypeConverter
    fun fromBytes(bytes: ByteArray?): FloatArray? {
        if (bytes == null) return null
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(buffer.remaining())
        buffer.get(out)
        return out
    }

    @TypeConverter
    fun toBytes(value: FloatArray?): ByteArray? {
        if (value == null) return null
        val buffer = ByteBuffer.allocate(value.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        buffer.asFloatBuffer().put(value)
        return buffer.array()
    }
}
