# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: config.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='config.proto',
  package='',
  syntax='proto2',
  serialized_options=b'\n\021myrtus.interfacesB\tEFCConfig',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0c\x63onfig.proto\"\x17\n\x06\x43onfig\x12\r\n\x05steps\x18\x01 \x02(\x03\x42\x1e\n\x11myrtus.interfacesB\tEFCConfig'
)




_CONFIG = _descriptor.Descriptor(
  name='Config',
  full_name='Config',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='steps', full_name='Config.steps', index=0,
      number=1, type=3, cpp_type=2, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=16,
  serialized_end=39,
)

DESCRIPTOR.message_types_by_name['Config'] = _CONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Config = _reflection.GeneratedProtocolMessageType('Config', (_message.Message,), {
  'DESCRIPTOR' : _CONFIG,
  '__module__' : 'config_pb2'
  # @@protoc_insertion_point(class_scope:Config)
  })
_sym_db.RegisterMessage(Config)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
