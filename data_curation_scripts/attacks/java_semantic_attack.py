import os
import re
import random
import json
from concurrent.futures import ProcessPoolExecutor
import sys

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    raise ImportError("nltk and wordnet required: pip install nltk && python -m nltk.downloader wordnet omw-1.4")

JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
    'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null', 'var'
}

COMMON_TYPE_NAMES = {
    'string', 'String', 'int', 'Integer', 'long', 'Long', 'float', 'Float',
    'double', 'Double', 'boolean', 'Boolean', 'char', 'Character', 'byte',
    'Byte', 'short', 'Short', 'object', 'Object', 'void', 'Void', 'list',
    'List', 'map', 'Map', 'set', 'Set', 'array', 'Array', 'dictionary',
    'Dictionary', 'arraylist', 'ArrayList', 'hashmap', 'HashMap', 'date',
    'Date', 'time', 'Time', 'exception', 'Exception', 'runtimeexception',
    'RuntimeException', 'system', 'System', 'console', 'Console', 'math',
    'Math', 'stringbuilder', 'StringBuilder', 'stringbuffer', 'StringBuffer',
    'scanner', 'Scanner', 'file', 'File', 'path', 'Path', 'stream', 'Stream',
    'iterator', 'Iterator', 'comparable', 'Comparable', 'comparator', 'Comparator',
    'serializable', 'Serializable', 'override', 'Override', 'deprecated',
    'Deprecated', 'nullable', 'Nullable', 'notnull', 'NotNull', 'data', 'Data',
    'entity', 'Entity', 'repository', 'Repository', 'service', 'Service',
    'controller', 'Controller', 'component', 'Component', 'configuration',
    'Configuration', 'autowired', 'Autowired', 'bean', 'Bean', 'value', 'Value',
    'requestmapping', 'RequestMapping', 'getmapping', 'GetMapping', 'postmapping',
    'PostMapping', 'json', 'JSON', 'xml', 'XML', 'http', 'HTTP', 'url', 'URL',
    'uri', 'URI', 'dto', 'DTO', 'dao', 'DAO', 'util', 'Util', 'utils', 'Utils',
    'helper', 'Helper', 'factory', 'Factory', 'builder', 'Builder', 'adapter',
    'Adapter', 'decorator', 'Decorator', 'observer', 'Observer', 'singleton',
    'Singleton', 'prototype', 'Prototype', 'strategy', 'Strategy', 'command',
    'Command', 'handler', 'Handler', 'processor', 'Processor', 'manager',
    'Manager', 'provider', 'Provider', 'consumer', 'Consumer', 'listener',
    'Listener', 'publisher', 'Publisher', 'subscriber', 'Subscriber',
    'context', 'Context', 'session', 'Session', 'request', 'Request',
    'response', 'Response', 'header', 'Header', 'cookie', 'Cookie',
    'principal', 'Principal', 'credential', 'Credential', 'token', 'Token',
    'jwt', 'JWT', 'oauth', 'OAuth', 'role', 'Role', 'authority', 'Authority',
    'permission', 'Permission', 'user', 'User', 'account', 'Account',
    'profile', 'Profile', 'order', 'Order', 'product', 'Product', 'item',
    'Item', 'cart', 'Cart', 'payment', 'Payment', 'invoice', 'Invoice',
    'customer', 'Customer', 'client', 'Client', 'supplier', 'Supplier',
    'employee', 'Employee', 'department', 'Department', 'project', 'Project',
    'task', 'Task', 'schedule', 'Schedule', 'calendar', 'Calendar',
    'notification', 'Notification', 'message', 'Message', 'email', 'Email',
    'sms', 'SMS', 'log', 'Log', 'logger', 'Logger', 'logging', 'Logging',
    'debug', 'Debug', 'info', 'Info', 'warn', 'Warn', 'error', 'Error',
    'fatal', 'Fatal', 'trace', 'Trace', 'assert', 'Assert', 'validate',
    'Validate', 'validation', 'Validation', 'verify', 'Verify', 'check',
    'Check', 'test', 'Test', 'testing', 'Testing', 'mock', 'Mock',
    'stub', 'Stub', 'spy', 'Spy', 'fixture', 'Fixture', 'benchmark',
    'Benchmark', 'performance', 'Performance', 'metric', 'Metric',
    'statistics', 'Statistics', 'analytics', 'Analytics', 'report',
    'Report', 'dashboard', 'Dashboard', 'widget', 'Widget', 'view',
    'View', 'model', 'Model', 'viewmodel', 'ViewModel', 'presenter',
    'Presenter', 'interactor', 'Interactor', 'router', 'Router',
    'navigator', 'Navigator', 'coordinator', 'Coordinator', 'delegate',
    'Delegate', 'datasource', 'DataSource', 'delegate', 'Delegate',
    'callback', 'Callback', 'closure', 'Closure', 'lambda', 'Lambda',
    'predicate', 'Predicate', 'function', 'Function', 'supplier',
    'Supplier', 'consumer', 'Consumer', 'bi', 'Bi', 'unary', 'Unary',
    'binary', 'Binary', 'ternary', 'Ternary', 'optional', 'Optional',
    'stream', 'Stream', 'collectors', 'Collectors', 'comparator',
    'Comparator', 'spliterator', 'Spliterator', 'base64', 'Base64',
    'uuid', 'UUID', 'regex', 'Regex', 'pattern', 'Pattern', 'matcher',
    'Matcher', 'charset', 'Charset', 'encoding', 'Encoding', 'locale',
    'Locale', 'timezone', 'TimeZone', 'instant', 'Instant', 'duration',
    'Duration', 'period', 'Period', 'localdate', 'LocalDate', 'localtime',
    'LocalTime', 'localdatetime', 'LocalDateTime', 'zoneddatetime',
    'ZonedDateTime', 'offsetdatetime', 'OffsetDateTime', 'year', 'Year',
    'month', 'Month', 'dayofweek', 'DayOfWeek', 'monthday', 'MonthDay',
    'yearmonth', 'YearMonth', 'clock', 'Clock', 'random', 'Random',
    'threadlocalrandom', 'ThreadLocalRandom', 'secure', 'Secure',
    'messageformat', 'MessageFormat', 'decimalformat', 'DecimalFormat',
    'numberformat', 'NumberFormat', 'dateformat', 'DateFormat',
    'simpledateformat', 'SimpleDateFormat', 'datetimeformatter',
    'DateTimeFormatter', 'format', 'Format', 'parse', 'Parse',
    'printer', 'Printer', 'parser', 'Parser', 'reader', 'Reader',
    'writer', 'Writer', 'inputstream', 'InputStream', 'outputstream',
    'OutputStream', 'bufferedreader', 'BufferedReader', 'bufferedwriter',
    'BufferedWriter', 'filereader', 'FileReader', 'filewriter', 'FileWriter',
    'printwriter', 'PrintWriter', 'bytearrayinputstream', 'ByteArrayInputStream',
    'bytearrayoutputstream', 'ByteArrayOutputStream', 'objectinputstream',
    'ObjectInputStream', 'objectoutputstream', 'ObjectOutputStream',
    'datainputstream', 'DataInputStream', 'dataoutputstream', 'DataOutputStream',
    'pipedinputstream', 'PipedInputStream', 'pipedoutputstream', 'PipedOutputStream',
    'sequenceinputstream', 'SequenceInputStream', 'pushbackinputstream',
    'PushbackInputStream', 'pushbackreader', 'PushbackReader', 'linenumberreader',
    'LineNumberReader', 'chartarrayreader', 'CharArrayReader', 'chararraywriter',
    'CharArrayWriter', 'stringreader', 'StringReader', 'stringwriter',
    'StringWriter', 'pipedreader', 'PipedReader', 'pipedwriter', 'PipedWriter',
    'filterreader', 'FilterReader', 'filterwriter', 'FilterWriter',
    'zipinputstream', 'ZipInputStream', 'zipoutputstream', 'ZipOutputStream',
    'gzipinputstream', 'GZIPInputStream', 'gzipoutputstream', 'GZIPOutputStream',
    'jarinputstream', 'JarInputStream', 'jaroutputstream', 'JarOutputStream',
    'inflaterinputstream', 'InflaterInputStream', 'deflateroutputstream',
    'DeflaterOutputStream', 'checkedinputstream', 'CheckedInputStream',
    'checkedoutputstream', 'CheckedOutputStream', 'digestinputstream',
    'DigestInputStream', 'digestoutputstream', 'DigestOutputStream',
    'cipherinputstream', 'CipherInputStream', 'cipheroutputstream',
    'CipherOutputStream', 'socket', 'Socket', 'serversocket', 'ServerSocket',
    'datagramsocket', 'DatagramSocket', 'multicastsocket', 'MulticastSocket',
    'urlconnection', 'URLConnection', 'httpurlconnection', 'HttpURLConnection',
    'ssl', 'SSL', 'tls', 'TLS', 'certificate', 'Certificate', 'keystore',
    'KeyStore', 'truststore', 'TrustStore', 'sslcontext', 'SSLContext',
    'sslsocketfactory', 'SSLSocketFactory', 'hostnameverifier', 'HostnameVerifier',
    'x509', 'X509', 'rsa', 'RSA', 'dsa', 'DSA', 'ec', 'EC', 'aes', 'AES',
    'des', 'DES', 'rc4', 'RC4', 'blowfish', 'Blowfish', 'sha', 'SHA',
    'md5', 'MD5', 'hmac', 'HMAC', 'pbkdf2', 'PBKDF2', 'bcrypt', 'BCrypt',
    'scrypt', 'SCrypt', 'argon2', 'Argon2', 'sql', 'SQL', 'jdbc', 'JDBC',
    'jpa', 'JPA', 'hibernate', 'Hibernate', 'mybatis', 'MyBatis', 'orm',
    'ORM', 'entitymanager', 'EntityManager', 'query', 'Query', 'criteria',
    'Criteria', 'transaction', 'Transaction', 'savepoint', 'Savepoint',
    'connection', 'Connection', 'datasource', 'DataSource', 'pool', 'Pool',
    'driver', 'Driver', 'statement', 'Statement', 'preparedstatement',
    'PreparedStatement', 'callablestatement', 'CallableStatement', 'resultset',
    'ResultSet', 'resultsetmetadata', 'ResultSetMetaData', 'databasemetadata',
    'DatabaseMetaData', 'rowset', 'RowSet', 'cachedrowset', 'CachedRowSet',
    'jdbctemplate', 'JdbcTemplate', 'namedparameterjdbctemplate',
    'NamedParameterJdbcTemplate', 'simplejdbctemplate', 'SimpleJdbcTemplate',
    'hibernatetemplate', 'HibernateTemplate', 'sessionfactory', 'SessionFactory',
    'transactionmanager', 'TransactionManager', 'platformtransactionmanager',
    'PlatformTransactionManager', 'jtatransactionmanager', 'JtaTransactionManager',
    'datasourcetransactionmanager', 'DataSourceTransactionManager',
    'hibernatetransactionmanager', 'HibernateTransactionManager',
    'resttemplate', 'RestTemplate', 'webclient', 'WebClient', 'httpheaders',
    'HttpHeaders', 'httpentity', 'HttpEntity', 'responseentity', 'ResponseEntity',
    'requestentity', 'RequestEntity', 'uricomponentsbuilder', 'UriComponentsBuilder',
    'uritemplate', 'UriTemplate', 'pathvariable', 'PathVariable', 'requestparam',
    'RequestParam', 'requestbody', 'RequestBody', 'responsebody', 'ResponseBody',
    'requestheader', 'RequestHeader', 'cookievalue', 'CookieValue',
    'modelattribute', 'ModelAttribute', 'sessionattribute', 'SessionAttribute',
    'initbinder', 'InitBinder', 'exceptionhandler', 'ExceptionHandler',
    'controlleradvice', 'ControllerAdvice', 'restcontrolleradvice',
    'RestControllerAdvice', 'enablewebmvc', 'EnableWebMvc', 'enablewebsecurity',
    'EnableWebSecurity', 'enableglobalmethodsecurity', 'EnableGlobalMethodSecurity',
    'preauthorize', 'PreAuthorize', 'postauthorize', 'PostAuthorize',
    'secured', 'Secured', 'rolesallowed', 'RolesAllowed', 'permitall',
    'PermitAll', 'denyall', 'DenyAll', 'authenticated', 'Authenticated',
    'fullyauthenticated', 'FullyAuthenticated', 'rememberme', 'RememberMe',
    'enablecaching', 'EnableCaching', 'cacheable', 'Cacheable', 'cacheevict',
    'CacheEvict', 'cacheput', 'CachePut', 'caching', 'Caching', 'cachmanager',
    'CacheManager', 'enableasync', 'EnableAsync', 'async', 'Async',
    'enablescheduling', 'EnableScheduling', 'scheduled', 'Scheduled',
    'enablebatchprocessing', 'EnableBatchProcessing', 'job', 'Job',
    'step', 'Step', 'itemreader', 'ItemReader', 'itemwriter', 'ItemWriter',
    'itemprocessor', 'ItemProcessor', 'jobrepository', 'JobRepository',
    'joblauncher', 'JobLauncher', 'jobexplorer', 'JobExplorer', 'partitioner',
    'Partitioner', 'tasklet', 'Tasklet', 'chunk', 'Chunk', 'skip',
    'Skip', 'retry', 'Retry', 'retryable', 'Retryable', 'recover',
    'Recover', 'circuitbreaker', 'CircuitBreaker', 'timelimiter',
    'TimeLimiter', 'ratelimiter', 'RateLimiter', 'bulkhead', 'Bulkhead',
    'enablecircuitbreaker', 'EnableCircuitBreaker', 'enableratelimiter',
    'EnableRateLimiter', 'enablebulkhead', 'EnableBulkhead', 'enableretry',
    'EnableRetry', 'feignclient', 'FeignClient', 'enablefeignclients',
    'EnableFeignClients', 'loadbalanced', 'LoadBalanced', 'ribbon',
    'Ribbon', 'eureka', 'Eureka', 'zuul', 'Zuul', 'gateway', 'Gateway',
    'hystrix', 'Hystrix', 'sleuth', 'Sleuth', 'zipkin', 'Zipkin',
    'configserver', 'ConfigServer', 'enableconfigserver', 'EnableConfigServer',
    'configclient', 'ConfigClient', 'refreshscope', 'RefreshScope',
    'bus', 'Bus', 'stream', 'Stream', 'binder', 'Binder', 'input',
    'Input', 'output', 'Output', 'processor', 'Processor', 'sink',
    'Sink', 'source', 'Source', 'aggregate', 'Aggregate', 'join',
    'Join', 'window', 'Window', 'tumbling', 'Tumbling', 'sliding',
    'Sliding', 'session', 'Session', 'event', 'Event', 'command',
    'Command', 'saga', 'Saga', 'orchestrator', 'Orchestrator',
    'choreography', 'Choreography', 'compensating', 'Compensating',
    'outbox', 'Outbox', 'inbox', 'Inbox', 'cdc', 'CDC', 'debezium',
    'Debezium', 'kafka', 'Kafka', 'rabbitmq', 'RabbitMQ', 'activemq',
    'ActiveMQ', 'jms', 'JMS', 'amqp', 'AMQP', 'mqtt', 'MQTT',
    'stomp', 'STOMP', 'websocket', 'WebSocket', 'sockjs', 'SockJS',
    'graphql', 'GraphQL', 'grpc', 'gRPC', 'thrift', 'Thrift',
    'protobuf', 'Protobuf', 'avro', 'Avro', 'jsonschema', 'JsonSchema',
    'openapi', 'OpenAPI', 'swagger', 'Swagger', 'postman', 'Postman',
    'selenium', 'Selenium', 'cucumber', 'Cucumber', 'gatling', 'Gatling',
    'jmeter', 'JMeter', 'sonarqube', 'SonarQube', 'pmd', 'PMD',
    'checkstyle', 'Checkstyle', 'spotbugs', 'SpotBugs', 'findbugs',
    'FindBugs', 'jacoco', 'JaCoCo', 'cobertura', 'Cobertura', 'pitest',
    'PITest', 'mutation', 'Mutation', 'staticanalysis', 'StaticAnalysis',
    'dynamicanalysis', 'DynamicAnalysis', 'profiling', 'Profiling',
    'instrumentation', 'Instrumentation', 'bytecode', 'Bytecode',
    'asm', 'ASM', 'cglib', 'CGLIB', 'javassist', 'Javassist',
    'bytebuddy', 'ByteBuddy', 'reflection', 'Reflection', 'proxy',
    'Proxy', 'invocationhandler', 'InvocationHandler', 'methodhandle',
    'MethodHandle', 'varhandle', 'VarHandle', 'lookup', 'Lookup',
    'callsite', 'CallSite', 'invokedynamic', 'InvokeDynamic',
    'constantpool', 'ConstantPool', 'classloader', 'ClassLoader',
    'urlclassloader', 'URLClassLoader', 'thread', 'Thread', 'runnable',
    'Runnable', 'callable', 'Callable', 'future', 'Future', 'completablefuture',
    'CompletableFuture', 'executor', 'Executor', 'executorservice',
    'ExecutorService', 'scheduledexecutorservice', 'ScheduledExecutorService',
    'threadpool', 'ThreadPool', 'forkjoinpool', 'ForkJoinPool',
    'forkjointask', 'ForkJoinTask', 'recursiveaction', 'RecursiveAction',
    'recursivetask', 'RecursiveTask', 'countdownlatch', 'CountDownLatch',
    'cyclicbarrier', 'CyclicBarrier', 'semaphore', 'Semaphore',
    'exchanger', 'Exchanger', 'phaser', 'Phaser', 'lock', 'Lock',
    'reentrantlock', 'ReentrantLock', 'reentrantreadwritelock',
    'ReentrantReadWriteLock', 'readlock', 'ReadLock', 'writelock',
    'WriteLock', 'condition', 'Condition', 'stampedlock', 'StampedLock',
    'atomic', 'Atomic', 'atomicinteger', 'AtomicInteger', 'atomiclong',
    'AtomicLong', 'atomicboolean', 'AtomicBoolean', 'atomicreference',
    'AtomicReference', 'atomicstampereference', 'AtomicStampedReference',
    'atomicmarkablereference', 'AtomicMarkableReference', 'atomicintegerarray',
    'AtomicIntegerArray', 'atomiclongarray', 'AtomicLongArray',
    'atomicreferencearray', 'AtomicReferenceArray', 'atomicintegerfieldupdater',
    'AtomicIntegerFieldUpdater', 'atomiclongfieldupdater', 'AtomicLongFieldUpdater',
    'atomicreferencefieldupdater', 'AtomicReferenceFieldUpdater',
    'longadder', 'LongAdder', 'longaccumulator', 'LongAccumulator',
    'doubleadder', 'DoubleAdder', 'doubleaccumulator', 'DoubleAccumulator',
    'accumulator', 'Accumulator', 'adder', 'Adder', 'striped', 'Striped',
    'weakreference', 'WeakReference', 'softreference', 'SoftReference',
    'phantomreference', 'PhantomReference', 'referencequeue', 'ReferenceQueue',
    'cleaner', 'Cleaner', 'finalize', 'Finalize', 'gc', 'GC',
    'runtime', 'Runtime', 'process', 'Process', 'processbuilder',
    'ProcessBuilder', 'inheritablethreadlocal', 'InheritableThreadLocal',
    'threadlocal', 'ThreadLocal', 'instrument', 'Instrument',
    'managementfactory', 'ManagementFactory', 'memorymxbean', 'MemoryMXBean',
    'memorypoolmxbean', 'MemoryPoolMXBean', 'garbagecollectormxbean',
    'GarbageCollectorMXBean', 'threadmxbean', 'ThreadMXBean',
    'compilationmxbean', 'CompilationMXBean', 'operatingsystemmxbean',
    'OperatingSystemMXBean', 'runtimeMXBean', 'RuntimeMXBean',
    'classloadingmxbean', 'ClassLoadingMXBean', 'notification',
    'Notification', 'notificationemitter', 'NotificationEmitter',
    'notificationlistener', 'NotificationListener', 'notificationfilter',
    'NotificationFilter', 'handback', 'Handback', 'mbean', 'MBean',
    'mxbean', 'MXBean', 'objectname', 'ObjectName', 'mbeanserver',
    'MBeanServer', 'mbeanserverconnection', 'MBeanServerConnection',
    'standardmbean', 'StandardMBean', 'dynamicmbean', 'DynamicMBean',
    'modelmbean', 'ModelMBean', 'openmbean', 'OpenMBean', 'composite',
    'Composite', 'tabular', 'Tabular', 'simpletype', 'SimpleType',
    'opentype', 'OpenType', 'descriptor', 'Descriptor', 'attribute',
    'Attribute', 'attributechange', 'AttributeChange', 'attributelist',
    'AttributeList', 'attributechangeemitter', 'AttributeChangeEmitter',
    'attributenotification', 'AttributeNotification', 'relation',
    'Relation', 'relationtype', 'RelationType', 'relationService',
    'RelationService', 'role', 'Role', 'roleresult', 'RoleResult',
    'rolenotfound', 'RoleNotFound', 'roleinfo', 'RoleInfo', 'agent',
    'Agent', 'jmx', 'JMX', 'rmi', 'RMI', 'iiop', 'IIOP', 'jndi',
    'JNDI', 'ldap', 'LDAP', 'corba', 'CORBA', 'idl', 'IDL', 'orb',
    'ORB', 'poa', 'POA', 'servant', 'Servant', 'tie', 'Tie', 'skel',
    'Skel', 'stub', 'Stub', 'registry', 'Registry', 'naming',
    'Naming', 'context', 'Context', 'initialcontext', 'InitialContext',
    'directory', 'Directory', 'dircontext', 'DirContext', 'attributes',
    'Attributes', 'basicattributes', 'BasicAttributes', 'searchcontrols',
    'SearchControls', 'searchresult', 'SearchResult', 'schema',
    'Schema', 'namingenumeration', 'NamingEnumeration', 'nameparser',
    'NameParser', 'compoundname', 'CompoundName', 'compositename',
    'CompositeName', 'reference', 'Reference', 'stringrefaddr',
    'StringRefAddr', 'binaryrefaddr', 'BinaryRefAddr', 'refaddr',
    'RefAddr', 'cannotproceed', 'CannotProceed', 'communicationexception',
    'CommunicationException', 'configurationexception', 'ConfigurationException',
    'contextnotempty', 'ContextNotEmpty', 'insufficientresources',
    'InsufficientResources', 'interruptednamingexception', 'InterruptedNamingException',
    'limitexceeded', 'LimitExceeded', 'linkexception', 'LinkException',
    'linkloop', 'LinkLoop', 'malformedlink', 'MalformedLink', 'namealreadybound',
    'NameAlreadyBound', 'namenotfound', 'NameNotFound', 'namingexception',
    'NamingException', 'namingsecurityexception', 'NamingSecurityException',
    'noinitialcontext', 'NoInitialContext', 'noPermission', 'NoPermission',
    'notcontext', 'NotContext', 'serviceunavailable', 'ServiceUnavailable',
    'sizeLimitExceeded', 'SizeLimitExceeded', 'timeLimitExceeded',
    'TimeLimitExceeded', 'authenticationexception', 'AuthenticationException',
    'authenticationnotsupported', 'AuthenticationNotSupported',
    'invalidattributes', 'InvalidAttributes', 'invalidattributestate',
    'InvalidAttributeState', 'invalidattributevalue', 'InvalidAttributeValue',
    'invalidsearchcontrols', 'InvalidSearchControls', 'invalidsearchfilter',
    'InvalidSearchFilter', 'schemaviolation', 'SchemaViolation',
    'namingevent', 'NamingEvent', 'naminglistener', 'NamingListener',
    'objectchange', 'ObjectChange', 'namespacechange', 'NamespaceChange',
    'eventcontext', 'EventContext', 'eventdircontext', 'EventDirContext',
    'spi', 'SPI', 'statefactory', 'StateFactory', 'objectfactory',
    'ObjectFactory', 'objectfactorybuilder', 'ObjectFactoryBuilder',
    'namingmanager', 'NamingManager', 'directorymanager', 'DirectoryManager',
    'resolver', 'Resolver', 'resolverresult', 'ResolverResult',
    'resolverprovider', 'ResolverProvider', 'dns', 'DNS', 'resolverstyle',
    'ResolverStyle', 'chronofield', 'ChronoField', 'chronounit', 'ChronoUnit',
    'temporal', 'Temporal', 'temporalaccessor', 'TemporalAccessor',
    'temporaladjuster', 'TemporalAdjuster', 'temporaladjusters',
    'TemporalAdjusters', 'temporalfield', 'TemporalField', 'temporalquery',
    'TemporalQuery', 'temporalqueries', 'TemporalQueries', 'temporalamount',
    'TemporalAmount', 'zoneid', 'ZoneId', 'zoneoffset', 'ZoneOffset',
    'zonerules', 'ZoneRules', 'isoregion', 'IsoRegion', 'weekfields',
    'WeekFields', 'formatstyle', 'FormatStyle', 'signstyle', 'SignStyle',
    'textstyle', 'TextStyle', 'resolverstyle', 'ResolverStyle',
    'parsed', 'Parsed', 'parsedleapsecond', 'ParsedLeapSecond',
    'dateTimeBuilder', 'DateTimeBuilder', 'dateTimePrintContext',
    'DateTimePrintContext', 'dateTimeParseContext', 'DateTimeParseContext',
    'decimalstyle', 'DecimalStyle', 'dateTimeFormatterBuilder',
    'DateTimeFormatterBuilder', 'reducedprinterparser', 'ReducedPrinterParser',
    'fractionprinterparser', 'FractionPrinterParser', 'literalprinterparser',
    'LiteralPrinterParser', 'numberprinterparser', 'NumberPrinterParser',
    'padprinterdecorator', 'PadPrinterDecorator', 'optionalprinterparser',
    'OptionalPrinterParser', 'settingsparser', 'SettingsParser',
    'compositeprinterparser', 'CompositePrinterParser', 'stringliteralprinterparser',
    'StringLiteralPrinterParser', 'zoneidprinterparser', 'ZoneIdPrinterParser',
    'zonetextprinterparser', 'ZoneTextPrinterParser', 'offsetidprinterparser',
    'OffsetIdPrinterParser', 'offsetprinterparser', 'OffsetPrinterParser',
    'localdatetimeprinterparser', 'LocalDateTimePrinterParser',
    'instantprinterparser', 'InstantPrinterParser', 'dateTimeTextProvider',
    'DateTimeTextProvider', 'simpleDateTimeTextProvider', 'SimpleDateTimeTextProvider'
}


STRING_PATTERNS = [
    (r'"(?:[^"\\\\]|\\\\.)*"', 'STRING'),
    (r"'(?:[^'\\\\]|\\\\.)*'", 'STRING'),
]
COMMENT_PATTERNS = [
    (r'//.*$', 'COMMENT'),
    (r'/\*.*?\*/', 'COMMENT'),
]

def split_camel_case(name):
    if re.match(r'^[A-Z_]+$', name):
        return [name.lower()]
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
    if not words:
        return [name.lower()]
    return [w.lower() for w in words]

def get_synonym(word, pos=wn.NOUN):
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None
    for syn in synsets:
        lemmas = [l.name().replace('_', ' ') for l in syn.lemmas()
                  if l.name().lower() != word.lower()]
        if lemmas:
            return random.choice(lemmas)
    return None

def to_camel_case(words, original_name):
    if not words:
        return original_name
    is_all_lower = original_name.islower()
    is_all_upper = original_name.isupper()
    starts_upper = original_name[0].isupper() if original_name else False

    if is_all_lower:
        return ''.join(w.lower() for w in words)
    if is_all_upper:
        return ''.join(w.upper() for w in words)
    result = []
    for i, w in enumerate(words):
        if i == 0 and not starts_upper:
            result.append(w.lower())
        else:
            result.append(w.capitalize())
    return ''.join(result)

def find_synonym_for_identifier(name):
    if not name or len(name) < 2:
        return None

    words = split_camel_case(name)
    new_words = []
    changed = False
    for w in words:
        syn = get_synonym(w, wn.NOUN) or get_synonym(w, wn.VERB)
        if syn:
            new_word = syn.replace(' ', '').replace('-', '')
            new_words.append(new_word)
            changed = True
        else:
            new_words.append(w)

    if not changed:
        return None

    return to_camel_case(new_words, name)

STRING_PATTERNS = [
    (r'"(?:[^"\\]|\\.)*"', 'STRING'),      # double-quoted strings
    (r"'(?:[^'\\]|\\.)*'", 'STRING'),      # single-quoted strings
]

COMMENT_PATTERNS = [
    (r'//.*$', 'COMMENT'),                    # single-line comments
    (r'/\*.*?\*/', 'COMMENT'),               # multi-line comments (non-greedy may fail across lines)
]

def tokenize_line(line):
    segments = []
    pos = 0
    n = len(line)

    while pos < n:
        matched = False
        for pattern, typ in COMMENT_PATTERNS:
            m = re.match(pattern, line[pos:])
            if m:
                segments.append((m.group(), typ))
                pos += m.end()
                matched = True
                break
        if matched:
            continue
        for pattern, typ in STRING_PATTERNS:
            m = re.match(pattern, line[pos:])
            if m:
                segments.append((m.group(), typ))
                pos += m.end()
                matched = True
                break
        if matched:
            continue
        code_start = pos
        while pos < n:
            found = False
            for pattern, typ in COMMENT_PATTERNS + STRING_PATTERNS:
                if re.match(pattern, line[pos:]):
                    found = True
                    break
            if found:
                break
            pos += 1

        if pos > code_start:
            segments.append((line[code_start:pos], 'CODE'))
        elif pos < n:
            segments.append((line[pos], 'CODE'))
            pos += 1

    return segments

def build_file_mapping(all_code_text, keywords_set):
    id_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    identifiers = set()
    for m in id_pattern.finditer(all_code_text):
        word = m.group()
        if (word not in keywords_set and
            word not in COMMON_TYPE_NAMES and
            len(word) >= 2):
            identifiers.add(word)

    mapping = {}
    for ident in identifiers:
        new_name = find_synonym_for_identifier(ident)
        if new_name and new_name != ident:
            mapping[ident] = new_name
    return mapping

def replace_identifiers_in_code(code_text, mapping):
    if not mapping:
        return code_text
    sorted_ids = sorted(mapping.keys(), key=len, reverse=True)
    escaped = [re.escape(i) for i in sorted_ids]
    replace_re = re.compile(r'\b(' + '|'.join(escaped) + r')\b')

    def replacer(m):
        return mapping[m.group(1)]

    return replace_re.sub(replacer, code_text)

def process_line(line, keywords_set, file_mapping):
    segments = tokenize_line(line)
    result = []
    for text, typ in segments:
        if typ == 'CODE':
            new_text = replace_identifiers_in_code(text, file_mapping)
            result.append(new_text)
        else:
            result.append(text)
    return ''.join(result)


def process_file_pair(args):
    f_in, f_out, mode = args
    try:
        with open(f_in, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        all_code = []
        for line in lines:
            segs = tokenize_line(line)
            for text, typ in segs:
                if typ == 'CODE':
                    all_code.append(text)
        all_code_text = ''.join(all_code)
        file_mapping = build_file_mapping(all_code_text, JAVA_KEYWORDS)
        modified = []
        for line in lines:
            modified.append(process_line(line, JAVA_KEYWORDS, file_mapping))
        if mode in ('structural', 'hybrid'):
            joined = ''.join(modified)
            if '{' in joined:
                joined = joined.replace('{', '{ { int _gate = 0; _gate++; } ', 1)
            modified = [joined]
        os.makedirs(os.path.dirname(f_out), exist_ok=True)
        with open(f_out, 'w', encoding='utf-8') as f:
            f.writelines(modified)
        return True
    except:
        return False

def run_attack(mode):
    print(f'--- Java Synonym Attack: Mode [{mode.upper()}] ---')
    output_dir = os.path.abspath(f'data/attack_v2__{mode}')
    smells = ['ComplexConditional', 'ComplexMethod', 'FeatureEnvy', 'MultifacetedAbstraction']
    with open('subset_files.json', 'r') as f:
        subset = json.load(f)['Java']
    for smell in smells:
        for case in ['Positive', 'Negative']:
            src = os.path.join('../data/java_subset_local', smell, case)
            dst = os.path.join(output_dir, smell, case)
            if not os.path.exists(src):
                continue
            os.makedirs(dst, exist_ok=True)
            target_list = subset[smell][case]
            print(f'   Streaming {len(target_list)} files for {smell}/{case}...')
            sys.stdout.flush()
            tasks = []
            for fname in target_list:
                f_in = os.path.join(src, fname)
                f_out = os.path.join(dst, fname.replace('.code', '.java'))
                if os.path.exists(f_out):
                    continue
                tasks.append((f_in, f_out, mode))
            with ProcessPoolExecutor(max_workers=5) as exe:
                for i, ok in enumerate(exe.map(process_file_pair, tasks)):
                    if (i + 1) % 500 == 0:
                        print(f'      - Processed {i+1}/{len(tasks)}...')
                        sys.stdout.flush()
            print(f'      - Done with {smell}/{case}')
            sys.stdout.flush()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_attack(sys.argv[1])
    else:
        for m in ['semantic', 'structural', 'hybrid']:
            run_attack(m)
